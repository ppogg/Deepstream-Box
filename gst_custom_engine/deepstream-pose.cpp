#include <nvdsgstutils.h>
#include <cuda_runtime_api.h>
#include <gstnvdsmeta.h>
#include <glib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <ctime>
#include <unordered_map>

#include "perf.h"

const gint skeleton[][2] = {
    {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13}, {6, 7}, {6, 8}, {7, 9}, {8, 10}, {9, 11}, {2, 3}, {1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}
    };

struct PoseData {
    gint video_width;
    gint video_height;
};

struct AppConfig {
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> config_map;
    
    void load_config(const std::string& config_file) {
        std::ifstream file(config_file);
        std::string line;
        std::string current_section;
        
        while (std::getline(file, line)) {
            line.erase(0, line.find_first_not_of(" \t"));
            line.erase(line.find_last_not_of(" \t") + 1);
            
            if (line.empty() || line[0] == '#') {
                continue;
            }
            
            if (line[0] == '[' && line.back() == ']') {
                current_section = line.substr(1, line.size() - 2);
            } else {
                size_t equals_pos = line.find('=');
                if (equals_pos != std::string::npos) {
                    std::string key = line.substr(0, equals_pos);
                    std::string value = line.substr(equals_pos + 1);
                    key.erase(key.find_last_not_of(" \t") + 1);
                    value.erase(0, value.find_first_not_of(" \t"));
                    config_map[current_section][key] = value;
                }
            }
        }
    }
    
    std::string get_string(const std::string& section, const std::string& key, const std::string& default_value = "") {
        auto section_it = config_map.find(section);
        if (section_it != config_map.end()) {
            auto key_it = section_it->second.find(key);
            if (key_it != section_it->second.end()) {
                return key_it->second;
            }
        }
        return default_value;
    }
    
    int get_int(const std::string& section, const std::string& key, int default_value = 0) {
        std::string value = get_string(section, key);
        if (!value.empty()) {
            return std::stoi(value);
        }
        return default_value;
    }
    
    float get_float(const std::string& section, const std::string& key, float default_value = 0.0f) {
        std::string value = get_string(section, key);
        if (!value.empty()) {
            return std::stof(value);
        }
        return default_value;
    }
    
    bool get_bool(const std::string& section, const std::string& key, bool default_value = false) {
        std::string value = get_string(section, key);
        if (!value.empty()) {
            return value == "1" || value == "true" || value == "True" || value == "TRUE";
        }
        return default_value;
    }
};

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer user_data)
{
    GMainLoop *loop = (GMainLoop *)user_data;
    switch (GST_MESSAGE_TYPE(msg))
    {
    case GST_MESSAGE_EOS:
    {
        g_print("Received EOS event...\n");
        g_print("Waiting for pipeline to finish processing...\n");

        GstState state;
        GstState pending;
        gst_element_get_state(GST_ELEMENT(GST_MESSAGE_SRC(msg)), &state, &pending, GST_CLOCK_TIME_NONE);

        g_print("Pipeline processing complete. Exiting...\n");

        g_main_loop_quit(loop);
        break;
    }
    case GST_MESSAGE_WARNING:
    {
        gchar *debug;
        GError *error;
        gst_message_parse_warning(msg, &error, &debug);
        g_printerr("WARNING: %s: %s\n", GST_OBJECT_NAME(msg->src), error->message);
        g_free(debug);
        g_error_free(error);
        break;
    }
    case GST_MESSAGE_ERROR:
    {
        gchar *debug;
        GError *error;
        gst_message_parse_error(msg, &error, &debug);
        g_printerr("ERROR: %s: %s\n", GST_OBJECT_NAME(msg->src), error->message);
        g_free(debug);
        g_error_free(error);
        g_main_loop_quit(loop);
        break;
    }
    default:
        break;
    }
    return TRUE;
}

static gboolean key_event(GIOChannel *source, GIOCondition condition, gpointer data)
{
    GMainLoop *loop = (GMainLoop *)data;
    g_main_loop_quit(loop);
    return FALSE;
}

std::string getCurrentTimeFilename(const char *file_format)
{
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::string file_format_name = file_format;
    std::tm tm_info;
    localtime_r(&now_time, &tm_info);
    std::ostringstream oss;
    oss << std::put_time(&tm_info, "%Y-%m-%d_%H-%M-%S");
    return "pose" + oss.str() + file_format_name;
}

static void pad_added_handler(GstElement *src, GstPad *new_pad, gpointer data)
{
    GstElement *streammux = (GstElement *)data;
    GstPad *sink_pad = gst_element_get_request_pad(streammux, "sink_0");
    GstPadLinkReturn ret;
    GstCaps *new_pad_caps = NULL;
    GstStructure *new_pad_struct = NULL;
    const gchar *new_pad_type = NULL;

    g_print("Received new pad '%s' from '%s':\n", GST_PAD_NAME(new_pad), GST_ELEMENT_NAME(src));

    new_pad_caps = gst_pad_get_current_caps(new_pad);
    new_pad_struct = gst_caps_get_structure(new_pad_caps, 0);
    new_pad_type = gst_structure_get_name(new_pad_struct);

    g_print("  Pad type: %s\n", new_pad_type);

    if (g_str_has_prefix(new_pad_type, "video/x-raw"))
    {
        ret = gst_pad_link(new_pad, sink_pad);
        if (GST_PAD_LINK_FAILED(ret))
        {
            g_print("  Link failed: %d\n", ret);
        }
        else
        {
            g_print("  Link succeeded\n");
        }
    }
    else
    {
        g_print("  Not a video pad, ignoring\n");
        gst_object_unref(sink_pad);
    }

    if (new_pad_caps != NULL)
    {
        gst_caps_unref(new_pad_caps);
    }
}

static void set_custom_bbox(NvDsObjectMeta *obj_meta)
{
    obj_meta->rect_params.border_color.alpha = 0.0;
    obj_meta->text_params.font_params.font_color.alpha = 0.0;
    obj_meta->text_params.text_bg_clr.alpha = 0.0;
}

static void parse_pose_from_meta(NvDsFrameMeta *frame_meta, NvDsObjectMeta *obj_meta, PoseData *pose_data)
{
    guint num_joints = obj_meta->mask_params.size / (sizeof(float) * 3);

    gfloat model_width = (gfloat)obj_meta->mask_params.width;
    gfloat model_height = (gfloat)obj_meta->mask_params.height;
    
    gfloat video_width = (gfloat)pose_data->video_width;
    gfloat video_height = (gfloat)pose_data->video_height;

    gfloat width_ratio = model_width / video_width;
    gfloat height_ratio = model_height / video_height;
    gfloat scale = MIN(width_ratio, height_ratio);
    
    gfloat scaled_width = video_width * scale;
    gfloat scaled_height = video_height * scale;
    gfloat pad_x = (model_width - scaled_width) / 2.0;
    gfloat pad_y = (model_height - scaled_height) / 2.0;

    NvDsBatchMeta *batch_meta = frame_meta->base_meta.batch_meta;
    NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
    nvds_add_display_meta_to_frame(frame_meta, display_meta);

    for (guint i = 0; i < num_joints; ++i)
    {
        gfloat xc = (obj_meta->mask_params.data[i * 3 + 0] - pad_x) / scale;
        gfloat yc = (obj_meta->mask_params.data[i * 3 + 1] - pad_y) / scale;

        if (display_meta->num_circles == MAX_ELEMENTS_IN_DISPLAY_META)
        {
            display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
            nvds_add_display_meta_to_frame(frame_meta, display_meta);
        }

        NvOSD_CircleParams *circle_params = &display_meta->circle_params[display_meta->num_circles];
        circle_params->xc = xc;
        circle_params->yc = yc;
        circle_params->radius = 5;
        circle_params->circle_color.red = 1.0;
        circle_params->circle_color.green = 1.0;
        circle_params->circle_color.blue = 1.0;
        circle_params->circle_color.alpha = 1.0;
        circle_params->has_bg_color = 1;
        circle_params->bg_color.red = 0.0;
        circle_params->bg_color.green = 0.0;
        circle_params->bg_color.blue = 1.0;
        circle_params->bg_color.alpha = 1.0;
        display_meta->num_circles++;
    }

    for (guint i = 0; i < sizeof(skeleton) / sizeof(skeleton[0]); ++i)
    {
        gint joint1 = skeleton[i][0] - 1;
        gint joint2 = skeleton[i][1] - 1;
        
        if (joint1 >= (int)num_joints || joint2 >= (int)num_joints)
            continue;
            
        gfloat x1 = (obj_meta->mask_params.data[joint1 * 3 + 0] - pad_x) / scale;
        gfloat y1 = (obj_meta->mask_params.data[joint1 * 3 + 1] - pad_y) / scale;
        gfloat x2 = (obj_meta->mask_params.data[joint2 * 3 + 0] - pad_x) / scale;
        gfloat y2 = (obj_meta->mask_params.data[joint2 * 3 + 1] - pad_y) / scale;

        if (display_meta->num_lines == MAX_ELEMENTS_IN_DISPLAY_META)
        {
            display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
            nvds_add_display_meta_to_frame(frame_meta, display_meta);
        }

        NvOSD_LineParams *line_params = &display_meta->line_params[display_meta->num_lines];
        line_params->x1 = x1;
        line_params->y1 = y1;
        line_params->x2 = x2;
        line_params->y2 = y2;
        line_params->line_width = 5;
        line_params->line_color.red = 0.0;
        line_params->line_color.green = 0.0;
        line_params->line_color.blue = 1.0;
        line_params->line_color.alpha = 1.0;
        display_meta->num_lines++;
    }

    g_free(obj_meta->mask_params.data);
    obj_meta->mask_params.width = 0;
    obj_meta->mask_params.height = 0;
    obj_meta->mask_params.size = 0;
}

static GstPadProbeReturn infer_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    
    PoseData *pose_data = (PoseData *)user_data;

    NvDsMetaList *l_frame = NULL;
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

        NvDsMetaList *l_obj = NULL;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);

            parse_pose_from_meta(frame_meta, obj_meta, pose_data);
            set_custom_bbox(obj_meta);
        }
    }

    return GST_PAD_PROBE_OK;
}

int main(int argc, char *argv[])
{
    std::string config_file = "pose.txt";
    if (argc > 1) {
        config_file = argv[1];
    }
    
    gst_init(&argc, &argv);
    
    AppConfig config;
    config.load_config(config_file);
    
    int SOURCE_TYPE = config.get_int("source", "type", 1);
    bool VIDEO_SAVE_MODE = config.get_bool("application", "video-save-mode", true);
    int SET_WIDTH = config.get_int("streammux", "width", 1280);
    int SET_HEIGHT = config.get_int("streammux", "height", 720);
    int STREAMMUX_TIMEOUT = config.get_int("streammux", "batched-push-timeout", 16000);
    int PERF_MEASUREMENT_INTERVAL_SEC = config.get_int("application", "perf-measurement-interval-sec", 1);
    
    std::string VIDEO_PATH = config.get_string("source", "uri", "../bodypose.mp4");
    std::string CONFIG_FILE_PATH = config.get_string("inference", "config-file-path", "../config_infer_cu.txt");
    
    PoseData pose_data;
    pose_data.video_width = SET_WIDTH;
    pose_data.video_height = SET_HEIGHT;

    GMainLoop *loop = g_main_loop_new(NULL, FALSE);

    GstElement *pipeline = gst_pipeline_new("test-pipeline");
    if (!pipeline)
    {
        g_printerr("Failed to create pipeline\n");
        return -1;
    }
    
    GstElement *source = NULL;
    GstElement *streammux = NULL;
    
    if (SOURCE_TYPE == 1)
    {
        g_print("Using file source mode\n");
        source = gst_element_factory_make("filesrc", "file-source");
        if (!source)
        {
            g_printerr("Failed to create file source element\n");
            return -1;
        }

        g_object_set(G_OBJECT(source), "location", VIDEO_PATH.c_str(), NULL);

        GstElement *decoder = gst_element_factory_make("decodebin", "decoder");
        if (!decoder)
        {
            g_printerr("Failed to create decoder element\n");
            return -1;
        }
        
        streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
        if (!streammux)
        {
            g_printerr("Failed to create streammux element\n");
            return -1;
        }

        g_object_set(G_OBJECT(streammux),
                     "width", SET_WIDTH,
                     "height", SET_HEIGHT,
                     "batch-size", config.get_int("streammux", "batch-size", 1),
                     "batched-push-timeout", STREAMMUX_TIMEOUT,
                     NULL);

        gst_bin_add_many(GST_BIN(pipeline), source, decoder, streammux, NULL);
        if (!gst_element_link_many(source, decoder, NULL))
        {
            g_printerr("Failed to link source and decoder\n");
            return -1;
        }
        g_signal_connect(decoder, "pad-added", G_CALLBACK(pad_added_handler), streammux);
    }
    else if (SOURCE_TYPE == 2)
    {
        g_print("Using camera source mode\n");
        source = gst_element_factory_make("nvarguscamerasrc", "camera-source");
        if (!source)
        {
            g_printerr("Failed to create camera source element\n");
            return -1;
        }

        g_object_set(G_OBJECT(source),
                     "sensor-id", config.get_int("camera", "sensor-id", 0),
                     "sensor-mode", config.get_int("camera", "sensor-mode", 2),
                     NULL);

        GstElement *csicaps = gst_element_factory_make("capsfilter", "csicapsfilter");

        GstCaps *caps = gst_caps_new_simple("video/x-raw(memory:NVMM)",
                                            "width", G_TYPE_INT, SET_WIDTH,
                                            "height", G_TYPE_INT, SET_HEIGHT,
                                            "framerate", GST_TYPE_FRACTION, 30, 1,
                                            NULL);
        g_object_set(G_OBJECT(csicaps), "caps", caps, NULL);
        gst_caps_unref(caps);
        
        streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
        if (!streammux)
        {
            g_printerr("Failed to create streammux element\n");
            return -1;
        }

        g_object_set(G_OBJECT(streammux),
                     "width", SET_WIDTH,
                     "height", SET_HEIGHT,
                     "batch-size", config.get_int("streammux", "batch-size", 1),
                     "batched-push-timeout", STREAMMUX_TIMEOUT,
                     NULL);

        gst_bin_add_many(GST_BIN(pipeline), source, csicaps, streammux, NULL);

        if (!gst_element_link(source, csicaps))
        {
            g_printerr("Failed to link source and caps\n");
            return -1;
        }
        
        GstPad *source_src_pad = gst_element_get_static_pad(csicaps, "src");
        GstPad *streammux_sink_pad = gst_element_get_request_pad(streammux, "sink_0");
        if (gst_pad_link(source_src_pad, streammux_sink_pad) != GST_PAD_LINK_OK)
        {
            g_printerr("Failed to link camera source and streammux\n");
            return -1;
        }
        gst_object_unref(source_src_pad);
        gst_object_unref(streammux_sink_pad);
    }
    else
    {
        g_printerr("Unknown source type: %d. Use type=1 for file or type=2 for camera.\n", SOURCE_TYPE);
        return -1;
    }

    GstElement *pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
    if (!pgie)
    {
        g_printerr("Failed to create pgie element\n");
        return -1;
    }

    g_object_set(G_OBJECT(pgie),
                 "config-file-path", CONFIG_FILE_PATH.c_str(),
                 "qos", config.get_int("inference", "qos", 0),
                 NULL);

    GstElement *nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideoconvert");
    if (!nvvidconv)
    {
        g_printerr("Failed to create nvvidconv element\n");
        return -1;
    }

    GstElement *osd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");
    if (!osd)
    {
        g_printerr("Failed to create osd element\n");
        return -1;
    }
    g_object_set(G_OBJECT(osd), "process-mode", MODE_GPU, "qos", config.get_int("inference", "qos", 0), NULL);
    
    gst_bin_add_many(GST_BIN(pipeline), pgie, nvvidconv, osd, NULL);
    
    if (!gst_element_link_many(streammux, pgie, nvvidconv, osd, NULL))
    {
        g_printerr("Failed to link streammux and other elements\n");
        return -1;
    }

    if (VIDEO_SAVE_MODE)
    {
        std::string mp4_filename = getCurrentTimeFilename(".mp4");
        GstElement *encoder = gst_element_factory_make("nvv4l2h264enc", "h264-encoder");
        if (!encoder)
        {
            g_printerr("Failed to create encoder element\n");
            return -1;
        }
        g_object_set(G_OBJECT(encoder),
                     "bitrate", config.get_int("encoder", "bitrate", 4000000),
                     "preset-level", config.get_int("encoder", "preset-level", 1),
                     NULL);

        GstElement *h264parse = gst_element_factory_make("h264parse", "h264-parser");
        if (!h264parse)
        {
            g_printerr("Failed to create h264parse element\n");
            return -1;
        }

        GstElement *qtmux = gst_element_factory_make("qtmux", "mp4-muxer");
        if (!qtmux)
        {
            g_printerr("Failed to create qtmux element\n");
            return -1;
        }

        g_object_set(G_OBJECT(qtmux),
                     "fragment-duration", config.get_int("muxer", "fragment-duration", 0),
                     "streamable", config.get_bool("muxer", "streamable", false),
                     "faststart", config.get_bool("muxer", "faststart", true),
                     NULL);

        GstElement *sink = gst_element_factory_make("filesink", "nvvideo-renderer");
        if (!sink)
        {
            g_printerr("Failed to create sink element\n");
            return -1;
        }
        g_object_set(G_OBJECT(sink), "location", mp4_filename.c_str(), NULL);

        gst_bin_add_many(GST_BIN(pipeline), encoder, h264parse, qtmux, sink, NULL);

        if (!gst_element_link_many(osd, encoder, h264parse, qtmux, sink, NULL))
        {
            g_printerr("Failed to link osd and sink elements\n");
            return -1;
        }
        
        g_print("Video will be saved to: %s\n", mp4_filename.c_str());
    }
    else
    {
        GstElement *sink = gst_element_factory_make("nv3dsink", "nvvideo-renderer");
        if (!sink)
        {
            g_printerr("Failed to create sink element\n");
            return -1;
        }
        g_object_set(G_OBJECT(sink), "async", 0, "sync", 0, "qos", 0, NULL);

        if (!gst_bin_add(GST_BIN(pipeline), sink))
        {
            g_printerr("Failed to add sink element to pipeline\n");
            return -1;
        }
        if (!gst_element_link(osd, sink))
        {
            g_printerr("Failed to link osd and sink\n");
            return -1;
        }
        
        g_print("Video will be displayed on screen\n");
    }

    GstBus *bus = gst_element_get_bus(pipeline);
    guint bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    GstPad *infer_src_pad = gst_element_get_static_pad(pgie, "src");
    if (!infer_src_pad)
    {
        g_printerr("ERROR: Failed to get infer src pad\n");
        return -1;
    }
    else
        gst_pad_add_probe(infer_src_pad, GST_PAD_PROBE_TYPE_BUFFER, infer_src_pad_buffer_probe, &pose_data, NULL);
    gst_object_unref(infer_src_pad);

    NvDsAppPerfStructInt *perf_struct = NULL;
    if (config.get_bool("application", "enable-perf-measurement", true))
    {
        GstPad *converter_sink_pad = gst_element_get_static_pad(nvvidconv, "sink");
        if (!converter_sink_pad)
        {
            g_printerr("ERROR: Failed to get converter sink pad\n");
            return -1;
        }
        else
        {
            perf_struct = (NvDsAppPerfStructInt *)g_malloc0(sizeof(NvDsAppPerfStructInt));
            enable_perf_measurement(perf_struct, converter_sink_pad, 1, PERF_MEASUREMENT_INTERVAL_SEC, 0, perf_cb);
        }
        gst_object_unref(converter_sink_pad);
    }

    gst_element_set_state(pipeline, GST_STATE_PAUSED);

    if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE)
    {
        g_printerr("ERROR: Failed to set pipeline to playing\n");
        return -1;
    }

    GIOChannel *io_stdin = g_io_channel_unix_new(fileno(stdin));
    g_io_add_watch(io_stdin, G_IO_IN, key_event, loop);
    g_print("Press ENTER to exit...\n");

    g_print("Running...\n");
    g_main_loop_run(loop);

    g_print("Stopping...\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    if (perf_struct) {
        g_free(perf_struct);
    }
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);

    return 0;
}