#include "util/string.h"
#include "visp/vision.h"

#include "testing.h"

namespace visp {

void compare_images(std::string_view name, image_view result, float tolerance = 0.01f) {
    path reference_path = test_dir().reference / name;
    path result_path = test_dir().results / name;

    image_save(result, result_path.string().c_str());
    image_data reference = image_load(reference_path.string().c_str());

    test_set_info(format(
        "while comparing images {} and {}", relative(result_path).string(),
        relative(reference_path).string()));
    test_with_tolerance with(tolerance);
    CHECK_IMAGES_EQUAL(result, reference);
}

VISP_BACKEND_TEST(test_mobile_sam)(backend_type bt) {
    path model_path = test_dir().models / "MobileSAM-F16.gguf";
    path input_path = test_dir().input / "cat-and-hat.jpg";

    backend_device b = backend_init(bt);
    sam_model model = sam_load_model(model_path.string().c_str(), b);
    image_data input = image_load(input_path.string().c_str());
    sam_encode(model, input);
    image_data mask_box = sam_compute(model, box_2d{{180, 110}, {505, 330}});
    image_data mask_point = sam_compute(model, i32x2{200, 300});

    char const* suffix = bt == backend_type::cpu ? "-cpu.png" : "-gpu.png";
    float tolerance = bt == backend_type::cpu ? 0.01f : 0.015f;
    compare_images(format("mobile_sam-box{}", suffix), mask_box, tolerance);
    compare_images(format("mobile_sam-point{}", suffix), mask_point, tolerance);
}

VISP_BACKEND_TEST(test_birefnet)(backend_type bt) {
    path model_path = test_dir().models / "BiRefNet-lite-F16.gguf";
    path input_path = test_dir().input / "wardrobe.jpg";
    std::string name = "birefnet";
    name += bt == backend_type::cpu ? "-cpu.png" : "-gpu.png";

    backend_device b = backend_init(bt);
    birefnet_model model = birefnet_load_model(model_path.string().c_str(), b);
    image_data input = image_load(input_path.string().c_str());
    image_data output = birefnet_compute(model, input);

    float tolerance = bt == backend_type::cpu ? 0.01f : 0.015f;
    compare_images(name, output, tolerance);
}

VISP_TEST(test_birefnet_dynamic) {
    path model_path = test_dir().models / "BiRefNet-dynamic-F16.gguf";
    if (!exists(model_path) || !backend_is_available(backend_type::gpu)) {
        throw test_skip{"Model not available"}; // it's a large model
    }
    // Test using 2 images with different resolutions one after the other
    path input_path1 = test_dir().input / "cat-and-hat.jpg";
    path input_path2 = test_dir().input / "wardrobe.jpg";

    backend_device b = backend_init(backend_type::gpu);
    birefnet_model model = birefnet_load_model(model_path.string().c_str(), b);
    image_data input1 = image_load(input_path1.string().c_str());
    image_data input2 = image_load(input_path2.string().c_str());
    image_data output1 = birefnet_compute(model, input1);
    image_data output2 = birefnet_compute(model, input2);

    compare_images("birefnet-dynamic.png", output2, 0.015f);
}

VISP_BACKEND_TEST(test_depth_anything)(backend_type bt) {
    path model_path = test_dir().models / "Depth-Anything-V2-Small-F16.gguf";
    path input_path = test_dir().input / "wardrobe.jpg";
    std::string name = "depth-anything";
    name += bt == backend_type::cpu ? "-cpu.png" : "-gpu.png";

    backend_device b = backend_init(bt);
    depthany_model model = depthany_load_model(model_path.string().c_str(), b);
    image_data input = image_load(input_path.string().c_str());
    image_data depth = depthany_compute(model, input);
    image_data output = image_f32_to_u8(depth, image_format::alpha_u8);

    float tolerance = bt == backend_type::cpu ? 0.01f : 0.015f;
    compare_images(name, output, tolerance);
}

VISP_BACKEND_TEST(test_depth_anything_3)(backend_type bt) {
    path model_path = test_dir().models / "Depth-Anything-3-Small-F16.gguf";
    path input_path = test_dir().input / "wardrobe.jpg";
    std::string suffix = bt == backend_type::cpu ? "-cpu.png" : "-gpu.png";
    // the GGUF is converted locally and the references are not on the CDN yet
    if (!exists(model_path) || !exists(test_dir().reference / ("depth-anything-3" + suffix))) {
        throw test_skip{"Model or reference not available"};
    }

    backend_device b = backend_init(bt);
    depthany3_model model = depthany3_load_model(model_path.string().c_str(), b);
    image_data input = image_load(input_path.string().c_str());
    depthany3_images result = depthany3_compute(model, input);

    image_data depth = image_normalize(result.depth);
    image_data confidence = image_normalize(result.confidence);

    float tolerance = bt == backend_type::cpu ? 0.01f : 0.015f;
    compare_images(
        "depth-anything-3" + suffix, image_f32_to_u8(depth, image_format::alpha_u8), tolerance);
    compare_images(
        "depth-anything-3-conf" + suffix, image_f32_to_u8(confidence, image_format::alpha_u8),
        tolerance);
}

VISP_BACKEND_TEST(test_migan)(backend_type bt) {
    path model_path = test_dir().models / "MIGAN-512-places2-F16.gguf";
    path image_path = test_dir().input / "bench-image.jpg";
    path mask_path = test_dir().input / "bench-mask.png";
    std::string name = "migan";
    name += bt == backend_type::cpu ? "-cpu.png" : "-gpu.png";

    backend_device b = backend_init(bt);
    migan_model model = migan_load_model(model_path.string().c_str(), b);
    image_data image = image_load(image_path.string().c_str());
    image_data mask = image_load(mask_path.string().c_str());
    image_data output = migan_compute(model, image, mask);
    image_data composited = image_alpha_composite(output, image, mask);

    compare_images(name, composited);
}

VISP_BACKEND_TEST(test_esrgan)(backend_type bt) {
    path model_path = test_dir().models / "RealESRGAN-x4plus_anime-6B-F16.gguf";
    path input_path = test_dir().input / "vase-and-bowl.jpg";
    std::string name = "esrgan";
    name += bt == backend_type::cpu ? "-cpu.png" : "-gpu.png";

    backend_device b = backend_init(bt);
    esrgan_model model = esrgan_load_model(model_path.string().c_str(), b);
    image_data input = image_load(input_path.string().c_str());
    image_data output = esrgan_compute(model, input);

    compare_images(name, output);
}

} // namespace visp