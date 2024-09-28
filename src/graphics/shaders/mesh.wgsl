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
    mip_bias: f32,
    // quality option to skip the cascade 0 raymarch done in this shader,
    // improving performance at the cost of worse looking light edges.
    // actually a bool but using that type here breaks alignment
    skip_raymarch: u32,
}
@group(1) @binding(0)
var<uniform> light_params: CascadeRenderParams;
@group(1) @binding(1)
var light_tex: texture_2d_array<f32>;
@group(1) @binding(2)
var cascade_tex: texture_2d<f32>;
@group(1) @binding(3)
var bilinear_samp: sampler;

struct Environment {
    ambient_light: vec3<f32>,
    // TODO: directional lights
}
@group(1) @binding(4)
var<uniform> environment: Environment;

// material

struct MaterialUniforms {
    base_color: vec4<f32>,
    // unlike PBR, the emissive color does nothing at the mesh shading stage.
    // it's used beforehand in the global illumination instead
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
const EPS: f32 = 1e-5;

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
    radiance: vec3<f32>,
    transparency: vec3<f32>,
}

// take a sample of the emission texture.
// automatically applies alpha, since it isn't used anywhere afterwards.
fn sample_emission(uv: vec2<f32>, mip_level: f32) -> vec3<f32> {
    let val = textureSampleLevel(
        light_tex,
        bilinear_samp,
        uv,
        0,
        mip_level,
    );
    return val.rgb * val.a;
}

// take a sample of the attenuation texture.
fn sample_attenuation(uv: vec2<f32>, mip_level: f32) -> vec4<f32> {
    return textureSampleLevel(
        light_tex,
        bilinear_samp,
        uv,
        1,
        mip_level,
    );
}

// raymarch on the light texture to gather radiance.
// this is copy-pasted from radiance_cascades.wgsl,
// see that file for a version with comments.
// TODO: share shader code using naga_oil
fn raymarch(ray: Ray) -> RayResult {
    var out: RayResult;

    let mip_level = max(0., light_params.mip_bias);
    let pixel_size = exp2(mip_level);
    let screen_size = vec2<f32>(textureDimensions(light_tex));

    out.transparency = vec3<f32>(1.);
    out.radiance = vec3<f32>(0.);

    var t = 0.;
    var t_step = pixel_size;
    var ray_pos = ray.start;
    let uv = ray_pos / screen_size;
    var prev_rad = sample_emission(uv, mip_level);
    var prev_attn = sample_attenuation(uv, mip_level);
    for (var loop_idx = 0u; loop_idx < 10000u; loop_idx++) {
        var range_overrun = false;
        if t_step > ray.range - t {
            t_step = ray.range - t;
            range_overrun = true;
        }

        t += t_step;
        ray_pos += t_step * ray.dir;

        if ray_pos.x < 0 || ray_pos.x >= screen_size.x || ray_pos.y < 0 || ray_pos.y >= screen_size.y {
            // left the screen.
            // skip the environment map here because the cascade range is very short,
            // assuming the upper cascade has sampled it already
            return out;
        }

        let uv = ray_pos / screen_size;
        let next_rad = sample_emission(uv, mip_level);
        let next_attn = sample_attenuation(uv, mip_level);
        let step_world = t_step * 0.1;

        let approx_attn = 0.5 * (next_attn + prev_attn);
        let mfp = approx_attn.a;
        let step_transp = select(
            vec3<f32>(0.),
            pow(approx_attn.rgb, vec3<f32>(step_world / mfp)),
            mfp > 0.,
        );
        let step_rad = 0.5 * step_world * (prev_rad * step_transp + next_rad);
        out.radiance += step_rad * out.transparency;
        out.transparency = out.transparency * step_transp;

        prev_rad = next_rad;
        prev_attn = next_attn;

        if range_overrun {
            return out;
        }

        if out.transparency.r < EPS && out.transparency.g < EPS && out.transparency.b < EPS {
            return out;
        }
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
    // (different clamp values from radiance_cascades.wgsl due to
    // multiplying by 0.5 at a different time)
    let br_edge = light_params.probe_count - vec2<u32>(1u);
    pos_probespace = clamp(
        pos_probespace,
        vec2<f32>(1.),
        vec2<f32>(br_edge) - vec2<f32>(1.),
    );
    // directions are arranged into four tiles, each taking a (0.5, 0.5) chunk of uv space
    let probe_uv = 0.5 * pos_probespace / vec2<f32>(textureDimensions(cascade_tex));

    // directions and radiances in order r, t, l, b
    var ray_dirs = array(
        vec2<f32>(1., 0.),
        vec2<f32>(0., -1.),
        vec2<f32>(-1., 0.),
        vec2<f32>(0., 1.),
    );
    // uv coordinates to add to probe_uv to sample the corresponding direction
    var sample_offsets = array(
        vec2<f32>(0., 0.),
        vec2<f32>(0.5, 0.5),
        vec2<f32>(0., 0.5),
        vec2<f32>(0.5, 0.),
    );
    var radiances = array(vec3<f32>(0.), vec3<f32>(0.), vec3<f32>(0.), vec3<f32>(0.));
    for (var i = 0u; i < 4u; i++) {
        var ray: Ray;
        ray.dir = ray_dirs[i];
        ray.start = in.clip_position.xy;
        ray.range = light_params.probe_range;
        if light_params.skip_raymarch != 0u {
            radiances[i] = textureSample(cascade_tex, bilinear_samp, probe_uv + sample_offsets[i]).rgb;
        } else {
            var rad = raymarch(ray);
            let is_opaque = rad.transparency.r < EPS && rad.transparency.g < EPS && rad.transparency.b < EPS;
            if !is_opaque {
                let next = textureSample(cascade_tex, bilinear_samp, probe_uv + sample_offsets[i]);
                rad.radiance += rad.transparency * next.rgb;
            }
            radiances[i] = rad.radiance;
        }
    }

    // each direction on the radiance probe covers a quarter segment of a 2-sphere,
    // and diffuse lighting is an integral over a hemisphere
    // centered on the surface normal.
    // approximate the integral by computing where the hemisphere's bottom plane
    // intersects with the vertical center planes of each probe quadrant
    // (this is hard to explain without being able to draw a picture..)
    var directions = array(
        vec3<f32>(1., 0., 0.),
        vec3<f32>(0., 1., 0.),
        vec3<f32>(-1., 0., 0.),
    );
    var irradiance = environment.ambient_light;
    var total_weight = 0.;
    for (var dir_idx = 0u; dir_idx < 2u; dir_idx++) {
        let dir = directions[dir_idx];
        let dir_normal = directions[dir_idx + 1u];
        let rad = radiances[dir_idx];
        let rad_opposite = radiances[dir_idx + 2u];

        let plane_isect = normalize(cross(dir_normal, normal));
        let angle = acos(plane_isect.z);
        let dir_coverage = select(angle, PI - angle, normal.z > 0.) / PI;
        let opposite_coverage = 1. - dir_coverage;
        // modify the coverage with a smoothstep to create a stronger effect.
        // this is completely nonphysical (but then again trying to fit 3D normals
        // into a 2D lighting system is pretty nonphysical to begin with)
        // and still does not match what a directional light in the xy plane would look like
        // (since that would have a cutoff at normal.z = 1,
        // whereas this one still has an effect past that),
        // but looks pretty good
        let dir_weight = smoothstep(0.25, 1., dir_coverage);
        let opposite_weight = smoothstep(0.25, 1., opposite_coverage);
        irradiance += dir_weight * rad + opposite_weight * rad_opposite;
        total_weight += dir_weight + opposite_weight;
    }
    // divide by the combined weight of all directions.
    // this is motivated by the idea that
    // if there's white light coming from all directions,
    // regardless of the normal we should get exactly (1, 1, 1) irradiance
    irradiance /= total_weight;

    return vec4<f32>(irradiance, 1.) * diffuse_color;
}

