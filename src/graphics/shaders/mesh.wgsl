//
// uniforms
//

// camera

struct CameraUniforms {
    view_proj: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

// lights

struct CascadeRenderParams {
    probe_spacing: f32,
    probe_range: f32,
    probe_count: vec2<u32>,
    // quality option to skip the cascade 0 raymarch done in this shader,
    // improving performance at the cost of worse looking light edges.
    // actually a bool but using that type here breaks alignment
    skip_raymarch: u32,
}
@group(1) @binding(0)
var<uniform> light_params: CascadeRenderParams;
@group(1) @binding(1)
var light_tex: texture_2d<f32>;
@group(1) @binding(2)
var cascade_tex: texture_2d<f32>;
@group(1) @binding(3)
var cascade_samp: sampler;

// material

struct MaterialUniforms {
    base_color: vec4<f32>,
    emissive_color: vec4<f32>,
}

@group(2) @binding(0)
var<uniform> material: MaterialUniforms;
@group(2) @binding(1)
var t_diffuse: texture_2d<f32>;
@group(2) @binding(2)
var s_diffuse: sampler;
@group(2) @binding(3)
var t_normal: texture_2d<f32>;
@group(2) @binding(4)
var s_normal: sampler;

// instance

struct InstanceUniforms {
    model: mat4x4<f32>,
}

@group(3) @binding(0)
var<uniform> instance: InstanceUniforms;

const SQRT_2: f32 = 1.41421562;
const PI: f32 = 3.14159265;
const HALF_PI: f32 = 1.5707963;
const PI_3_2: f32 = 4.71238898;

//
// vertex shader
//

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
};

// counteract the scaling effect of a transformation
// in order to transform normals correctly
fn mat3_inv_scale_sq(m: mat3x3<f32>) -> vec3<f32> {
    return vec3<f32>(
        1.0 / dot(m[0].xyz, m[0].xyz),
        1.0 / dot(m[1].xyz, m[1].xyz),
        1.0 / dot(m[2].xyz, m[2].xyz)
    );
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
) -> VertexOutput {
    var out: VertexOutput;

    let model = instance.model;

    let pos_world = model * vec4<f32>(position, 1.);
    let model_3 = mat3x3<f32>(model[0].xyz, model[1].xyz, model[2].xyz);
    let inv_scaling = mat3_inv_scale_sq(model_3);
    let norm_transformed = inv_scaling * (model_3 * normal);
    let tan_transformed = inv_scaling * (model_3 * tangent);

    out.clip_position = camera.view_proj * pos_world;
    out.world_position = pos_world.xyz;
    out.tex_coords = tex_coords;
    out.normal = normalize(norm_transformed);
    out.tangent = normalize(tan_transformed);

    return out;
}

//
// fragment shader
//


// the final radiance cascade is done during shading,
// essentially placing a probe at each rendered pixel.
// this increases quality and reduces memory requirements
// but somewhat increases the cost of pixel shading.
// this is a simplified version of the `raymarch` function in radiance_cascades.wgsl,
// only raymarching on the 0th mip level
// and ignoring translucent materials for simplicity,
// assuming the drop in quality is minimal due to short range

struct Ray {
    start: vec2<f32>,
    dir: vec2<f32>,
    range: f32,
}

struct RayResult {
    value: vec3<f32>,
    is_radiance: bool,
}

fn raymarch(ray: Ray) -> RayResult {
    var out: RayResult;

    var screen_size = vec2<i32>(textureDimensions(light_tex));

    var t = 0.;
    var ray_pos = ray.start;
    var pixel_pos = vec2<i32>(ray_pos);
    let pixel_dir = vec2<i32>(sign(ray.dir));
    // bounded loop as a failsafe to avoid hanging
    // in case there's a bug that causes the raymarch to stop in place
    for (var loop_idx = 0u; loop_idx < 10000u; loop_idx++) {
        if pixel_pos.x < 0 || pixel_pos.x >= screen_size.x || pixel_pos.y < 0 || pixel_pos.y >= screen_size.y {
            // left the screen
            // just treat the edge of the screen as a shadow for now,
            // TODO: return radiance from an environment map
            out.value = vec3<f32>(0.);
            out.is_radiance = true;
            return out;
        }

        if t > ray.range {
            out.is_radiance = false;
            return out;
        }

        let rad = textureLoad(light_tex, pixel_pos, 0);
        if rad.a == 1. {
            out.value = rad.rgb;
            out.is_radiance = true;
            return out;
        }

        // move to the next pixel intersected by the ray.
        // simplifying assumption: we started at the center of a pixel 
        // and move only in diagonal directions,
        // hence being able to skip across corners instead of moving one axis at a time.
        // this also reduces texture loads by a ~third which is nice for perf
        let x_threshold = f32(select(pixel_pos.x, pixel_pos.x + 1, pixel_dir.x == 1));
        let t_step = abs((x_threshold - ray_pos.x) / ray.dir.x);
        t += t_step;
        ray_pos += t_step * ray.dir;
        pixel_pos += pixel_dir;
    }

    return out;
}

@fragment
fn fs_main(
    in: VertexOutput
) -> @location(0) vec4<f32> {
    // get the necessary parameters

    let diffuse_color = material.base_color * textureSample(t_diffuse, s_diffuse, in.tex_coords);

    let bitangent = cross(in.tangent, in.normal);
    let tbn = mat3x3(in.tangent, bitangent, in.normal);

    let tex_normal = textureSample(t_normal, s_normal, in.tex_coords).xyz;
    let normal = tbn * normalize(tex_normal * 2. - 1.);

    // look up the nearest radiance probe and compute lighting based on it

    // -0.5 because probe positioning is offset from the corner by half a space
    var pos_probespace = (in.clip_position.xy / light_params.probe_spacing) - vec2<f32>(0.5);
    // clamp to avoid interpolation getting values from adjacent tiles
    pos_probespace = clamp(
        vec2<f32>(0.5),
        vec2<f32>(light_params.probe_count) - vec2<f32>(0.5),
        pos_probespace,
    );
    // directions are arranged into four tiles, each taking a (0.5, 0.5) chunk of uv space
    let probe_uv = 0.5 * pos_probespace / vec2<f32>(light_params.probe_count);

    // directions and radiances in order tr, tl, bl, br
    var ray_dirs = array(
        vec2<f32>(SQRT_2, -SQRT_2),
        vec2<f32>(-SQRT_2, -SQRT_2),
        vec2<f32>(-SQRT_2, SQRT_2),
        vec2<f32>(SQRT_2, SQRT_2),
    );
    // uv coordinates to add to probe_uv to sample the corresponding direction
    var sample_offsets = array(
        vec2<f32>(0.5, 0.5),
        vec2<f32>(0., 0.5),
        vec2<f32>(0.5, 0.),
        vec2<f32>(0., 0.),
    );
    var radiances = array(vec3<f32>(0.), vec3<f32>(0.), vec3<f32>(0.), vec3<f32>(0.));
    for (var i = 0u; i < 4u; i++) {
        var ray: Ray;
        ray.dir = ray_dirs[i];
        ray.start = in.clip_position.xy;
        ray.range = light_params.probe_range;
        if light_params.skip_raymarch != 0u {
            radiances[i] = textureSample(cascade_tex, cascade_samp, probe_uv + sample_offsets[i]).rgb;
        } else {
            var rad = raymarch(ray);
            if !rad.is_radiance {
                let next = textureSample(cascade_tex, cascade_samp, probe_uv + sample_offsets[i]);
                rad.value = next.rgb;
            }
            radiances[i] = rad.value;
        }
    }

    // each direction on the radiance probe covers a quarter segment of a 2-sphere,
    // and diffuse lighting is an integral over a hemisphere
    // centered on the surface normal.
    // approximate the integral by computing where the hemisphere's bottom plane
    // intersects with the vertical center planes of each probe quadrant
    // (this is hard to explain without being able to draw a picture..)
    var directions = array(
        vec3<f32>(SQRT_2, SQRT_2, 0.),
        vec3<f32>(-SQRT_2, SQRT_2, 0.),
        vec3<f32>(-SQRT_2, -SQRT_2, 0.),
    );
    var radiance = vec3<f32>(0.);
    for (var dir_idx = 0u; dir_idx < 2u; dir_idx++) {
        let dir = directions[dir_idx];
        let dir_normal = directions[dir_idx + 1u];
        let rad = radiances[dir_idx];
        let rad_opposite = radiances[dir_idx + 2u];

        let plane_isect = normalize(cross(dir_normal, normal));
        let angle = acos(plane_isect.z);
        let dir_coverage = select(angle, PI - angle, normal.z > 0.) / PI;
        radiance += dir_coverage * rad + (1. - dir_coverage) * rad_opposite;
    }

    return vec4<f32>(radiance, 1.) * diffuse_color;
}

