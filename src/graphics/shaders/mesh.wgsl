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

struct DirectionalLight {
    direct_color: vec3<f32>,
    ambient_color: vec3<f32>,
    // the w component determines whether or not the direction
    // is actually used in shading;
    // if 1 we do normal shading and if 0 we only do a flat ambient light
    direction: vec4<f32>,
}

struct PointLight {
    position: vec3<f32>,
    color: vec3<f32>,
    radius: f32,
    attn_linear: f32,
    attn_quadratic: f32,
}

struct PointLights {
    count: u32,
    lights: array<PointLight, 1024>,
}

@group(1) @binding(0)
var<uniform> dir_light: DirectionalLight;
@group(1) @binding(1)
var<storage> point_lights: PointLights;

// joints

@group(2) @binding(0)
var<storage> joint_mats: array<mat4x4<f32>>;

// material

struct MaterialUniforms {
    base_color: vec4<f32>,
}

@group(3) @binding(0)
var<uniform> material: MaterialUniforms;
@group(3) @binding(1)
var t_diffuse: texture_2d<f32>;
@group(3) @binding(2)
var s_diffuse: sampler;
@group(3) @binding(3)
var t_normal: texture_2d<f32>;
@group(3) @binding(4)
var s_normal: sampler;

// instance

struct InstanceUniforms {
    model_row0: vec4<f32>,
    model_row1: vec4<f32>,
    model_row2: vec4<f32>,
    joint_offset: u32,
}

@group(4) @binding(0)
var<uniform> instance: InstanceUniforms;

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

// vertex shader with skinning, joints and weights in a separate vertex buffer
@vertex
fn vs_skinned(
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    // additional vertex data for skinning in a separate buffer
    // (u16 not supported in wgsl, so bit-twiddle joint indices from two u32s)
    @location(2) joints: vec2<u32>,
    @location(3) weights: vec4<f32>,
) -> VertexOutput {
    var out: VertexOutput;

    let model = mat4x4<f32>(
        instance.model_row0.x, instance.model_row1.x, instance.model_row2.x, 0.,
        instance.model_row0.y, instance.model_row1.y, instance.model_row2.y, 0.,
        instance.model_row0.z, instance.model_row1.z, instance.model_row2.z, 0.,
        instance.model_row0.w, instance.model_row1.w, instance.model_row2.w, 1.,
    );

    let joint_indices = vec4<u32>(instance.joint_offset) + vec4<u32>(
        joints[0] & 0xFFFFu,
        joints[0] >> 16u,
        joints[1] & 0xFFFFu,
        joints[1] >> 16u
    );

    var pos = vec3<f32>(0.);
    var has_joints = false;
    // hardcoded normal and tangent in the x,y plane,
    // since we don't support general 3D rendering
    let normal = vec3<f32>(0., 0., -1.);
    let tangent = vec3<f32>(1., 0., 0.);
    var norm_skinned = vec3<f32>(0.);
    var tan_skinned = vec3<f32>(0.);

    for (var i = 0; i < 4; i++) {
        let weight = weights[i];
        if weight > 0. {
            has_joints = true;
            let ji = joint_indices[i];
            let joint_mat = joint_mats[ji];
            pos += weight * (joint_mat * vec4<f32>(position, 1.)).xyz;

            let joint_mat_3 = mat3x3<f32>(joint_mat[0].xyz, joint_mat[1].xyz, joint_mat[2].xyz);
            let inv_scaling = mat3_inv_scale_sq(joint_mat_3);
            let weight_scaled = weight * inv_scaling;
            norm_skinned += weight_scaled * (joint_mat_3 * normal);
            tan_skinned += weight_scaled * (joint_mat_3 * tangent);
        }
    }
    // if no joints had any weight, fallback to original values
    if !has_joints {
        pos = position;
        norm_skinned = normal;
        tan_skinned = tangent;
    }

    // transform skinned values with the model matrix
    let pos_model = model * vec4<f32>(pos, 1.);
    let model_3 = mat3x3<f32>(model[0].xyz, model[1].xyz, model[2].xyz);
    let inv_scaling = mat3_inv_scale_sq(model_3);
    norm_skinned = inv_scaling * (model_3 * norm_skinned);
    tan_skinned = inv_scaling * (model_3 * tan_skinned);

    out.clip_position = camera.view_proj * pos_model;
    out.world_position = pos_model.xyz;
    out.tex_coords = tex_coords;
    out.normal = normalize(norm_skinned);
    out.tangent = normalize(tan_skinned);

    return out;
}

// vertex shader without skinning
@vertex
fn vs_unskinned(
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;

    let model = mat4x4<f32>(
        instance.model_row0.x, instance.model_row1.x, instance.model_row2.x, 0.,
        instance.model_row0.y, instance.model_row1.y, instance.model_row2.y, 0.,
        instance.model_row0.z, instance.model_row1.z, instance.model_row2.z, 0.,
        instance.model_row0.w, instance.model_row1.w, instance.model_row2.w, 1.,
    );

    let normal = vec3<f32>(0., 0., -1.);
    let tangent = vec3<f32>(1., 0., 0.);

    let pos_model = model * vec4<f32>(position, 1.);
    let model_3 = mat3x3<f32>(model[0].xyz, model[1].xyz, model[2].xyz);
    let inv_scaling = mat3_inv_scale_sq(model_3);
    let norm_transformed = inv_scaling * (model_3 * normal);
    let tan_transformed = inv_scaling * (model_3 * tangent);

    out.clip_position = camera.view_proj * pos_model;
    out.world_position = pos_model.xyz;
    out.tex_coords = tex_coords;
    out.normal = normalize(norm_transformed);
    out.tangent = normalize(tan_transformed);

    return out;
}

//
// fragment shader
//

@fragment
fn fs_main(
    in: VertexOutput
) -> @location(0) vec4<f32> {
    // color texture and normal map

    let diffuse_color = material.base_color * textureSample(t_diffuse, s_diffuse, in.tex_coords);

    let bitangent = cross(in.tangent, in.normal);
    let tbn = mat3x3(in.tangent, bitangent, in.normal);

    let tex_normal = textureSample(t_normal, s_normal, in.tex_coords).xyz;
    let normal = tbn * normalize(tex_normal * 2. - 1.);

    // directional light

    var diffuse_light: vec3<f32>;
    var ambient_light: vec3<f32>;
    if dir_light.direction.w == 0. {
        // no direct light, flat ambient lighting only
        diffuse_light = vec3<f32>(0., 0., 0.);
        ambient_light = dir_light.ambient_color;
    } else {
        // dot with the negative light direction
        // indicates how opposite to the light the normal is,
        // and hence the strength of the diffuse light
        let normal_dot_light = -dot(normal, dir_light.direction.xyz);

        let diffuse_strength = max(normal_dot_light, 0.);
        diffuse_light = diffuse_strength * dir_light.direct_color;

        // stylized approximation: ambient light comes from the direction opposite to the main light
        // TODO: instead of hardcoding intensity 0.1 here,
        // give it as part of the ambient color
        let ambient_strength = 0.1 + 0.1 * max(-normal_dot_light, 0.);
        ambient_light = dir_light.ambient_color * ambient_strength;
    }

    // point lights

    var point_light_total = vec3<f32>(0., 0., 0.);
    for (var li: u32 = 0u; li < point_lights.count; li++) {
        let light = point_lights.lights[li];

        let from_light = in.world_position - light.position;
        let dist = length(from_light);
        let attenuation = 1. / (1. + dist * light.attn_linear + dist * dist * light.attn_quadratic);

        let light_dir = from_light / dist;
        let normal_dot_light = -dot(normal, light_dir);

        let light_strength = attenuation * max(normal_dot_light, 0.);
        let light_contrib = light_strength * light.color;
        point_light_total += light_contrib;
    }

    let full_color = vec4<f32>(ambient_light + diffuse_light + point_light_total, 1.) * diffuse_color;
    return full_color;
}

