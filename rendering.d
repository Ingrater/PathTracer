module rendering;

import config;
import camera;
import scene;

import thBase.math3d.all;
import thBase.io;
import std.random;
import std.math;
import core.stdc.math;
import thBase.container.vector;
import thBase.container.hashmap;
import thBase.policies.hashing;
import thBase.math : max;

// global variables
__gshared Camera g_camera;
__gshared Scene g_scene;
__gshared uint g_width = 512;
__gshared uint g_height = 512;
__gshared uint g_numThreads = 8;

// Holds all information for a single pixel visible on the screen
struct Pixel
{
  vec3 color; //resulting pixel color. Each component has to be in the range [0,1]
  // Define additional per pixel values here
  vec3 normal;   // normal in world space
  vec3 position; // position in world space 
  vec3 directLight; // direct light from static light sources
  float n = 0.0f;
  vec3 sum;
  vec2[numSamples] samples;
  bool rastered = false;
  uint numSkippedSamples = 0;
  uint ambient = 0;
};

// Holds all information about a material
struct Material
{
  float emissive;
  vec3 color;
};

// Holds all information for a emissive triangle
struct LightTriangle
{
  vec3 v0; // one vertex of the triangle
  vec3 e1, e2; //the two edges of the triangle
  float area; // the area of the light triangle
  float probability;
}
__gshared Vector!LightTriangle[] g_lightTriangles;
__gshared float g_totalLightSourceArea[];

// information about sun
__gshared vec3 g_sunDir;
__gshared vec3 g_sunRadiance;
__gshared vec3 g_skyRadiance;
__gshared bool g_hasSky = false;
__gshared bool g_hasLights = true;

// Gets called on program shutdown
shared static ~this()
{
  Delete(g_camera);
  Delete(g_scene);
  foreach(ref light; g_lightTriangles)
    Delete(light);
  Delete(g_lightTriangles);
  Delete(g_totalLightSourceArea);
}


// Loads the scene and creates a camera
void loadScene()
{
  g_sunDir = vec3(1, 1, 3).normalized();
  g_sunRadiance = vec3(1.0f, 0.984f, 0.8f) * 10.0f;
  g_skyRadiance = vec3(0.682f, 0.977f, 1.0f);

  g_camera = New!Camera(30.0f, cast(float)g_height / cast(float)g_width);

  g_camera.setTransform(vec3(-660, -350, 600), vec3(-658, -349, 599.8), vec3(0, 0, 1));
  //g_scene = New!Scene("cornell-box-textured.thModel", &fillMaterial, mat4.Identity);
  //g_scene = New!Scene("citymap.thModel", &fillMaterial, ScaleMatrix(0.1f, 0.1f, 0.1f));
  //g_scene.saveTree("citymap.tree");
  g_scene = New!Scene("citymap.tree", &fillMaterial, mat4.Identity);
  //g_scene = New!Scene("citymapLights.tree", &fillMaterial, mat4.Identity);

  //find all light triangles
  uint numLightMaterials = 0;
  auto matMap = composite!(Hashmap!(const(Material)*, uint, PointerHashPolicy))(defaultCtor);
  foreach(ref mat; g_scene.materials)
  {
    if(mat.emissive > 0.0f)
    {
      matMap[&mat] = numLightMaterials;
      numLightMaterials++;
    }
  }

  auto lights = NewArray!(Vector!LightTriangle)(numLightMaterials);
  float[] totalLightSourceArea = NewArray!float(numLightMaterials);
  totalLightSourceArea[] = 0.0f;
  foreach(ref lightTriangles; lights)
  {
    lightTriangles = New!(Vector!LightTriangle)();
  }
  foreach(size_t i, ref triangleData; g_scene.triangleData)
  {
    if(triangleData.material !is null && triangleData.material.emissive > 0.0f)
    {
      assert(matMap.exists(triangleData.material));
      auto lightIndex = matMap[triangleData.material];
      LightTriangle lightTriangle;
      const(Triangle)* t = &g_scene.triangles[i];
      lightTriangle.v0 = t.v0;
      lightTriangle.e1 = t.v1 - t.v0;
      lightTriangle.e2 = t.v2 - t.v0;
      lightTriangle.area = t.area;
      lights[lightIndex] ~= lightTriangle;
      totalLightSourceArea[lightIndex] += lightTriangle.area;
    }
  }
  g_totalLightSourceArea = totalLightSourceArea;
  // compute probabilities
  foreach(size_t i, ref lightTriangles; lights)
  {
    foreach(ref light; lightTriangles)
    {
      light.probability = light.area / totalLightSourceArea[i];
    }
    writefln("Light %d has %d light triangles", i, lightTriangles.length);
  }
  g_lightTriangles = lights;
}

vec3 pickRandomLightPoint(ref Random gen, uint lightIndex)
{
  float a = uniform(0.0f, 1.0f, gen);
  foreach(ref light; g_lightTriangles[lightIndex])
  {
    if(light.probability > a)
    {
      while(true)
      {
        float u = uniform(0.0f, 1.0f, gen);
        float v = uniform(0.0f, 1.0f, gen);
        if(u + v <= 1.0f)
        {
          return light.v0 + light.e1 * v + light.e2 * u;
        }
      }
    }
    a -= light.probability;
  }
  auto light = g_lightTriangles[lightIndex][g_lightTriangles[lightIndex].length-1];
  while(true)
  {
    float u = uniform(0.0f, 1.0f, gen);
    float v = uniform(0.0f, 1.0f, gen);
    if(u + v <= 1.0f)
    {
      return light.v0 + light.e1 * v + light.e2 * u;
    }
  }
}

// Called for each material found in the model file
void fillMaterial(ref Material mat, const(char)[] materialName)
{
  writefln("%s", materialName);
  mat.color.x = 0.7f;
  mat.color.y = 0.7f;
  mat.color.z = 0.7f;
  mat.emissive = 0.0f;
  if(materialName == "Material #10088/City_HotelLight_mat")
  {
    mat.color.x = 0.7f;
    mat.color.y = 0.7f;
    mat.color.z = 0.7f;
    mat.emissive = 10.0f;
  }
  else if(materialName == "Material #10088/GreenLight")
  {
    mat.color.x = 0.0f;
    mat.color.y = 0.7f;
    mat.color.z = 0.0f;
    mat.emissive = 10.0f;
  }
  else if(materialName == "Material #10088/City_CinemaDisplay_mat")
  {
    mat.color.x = 0.7f;
    mat.color.y = 0.7f;
    mat.color.z = 0.7f;
    mat.emissive = 20.0f;
  }
  else if(materialName == "Material #10088/City_CinemaEntrancelamp_mat")
  {
    mat.color.x = 0.8f;
    mat.color.y = 0.8f;
    mat.color.z = 0.2f;
    mat.emissive = 160.0f;
  }
  else if(materialName == "Material #10088/City_FC_Light_mat")
  {
    mat.color.x = 0.7f;
    mat.color.y = 0.0f;
    mat.color.z = 0.0F;
    mat.emissive = 40.0f;
  }
  else if(materialName == "Material #10088/City_FC_Logo_mat")
  {
    mat.color.x = 0.7f;
    mat.color.y = 0.0f;
    mat.color.z = 0.0F;
    mat.emissive = 30.0f;
  }
}

// Computes a view ray for a given pixel index
Ray getViewRay(uint pixelIndex, ref Random gen)
{
  float pixelSizeX = 1.0f / cast(float)g_width;
  float pixelSizeY = 1.0f / cast(float)g_height;
  float x = cast(float)(pixelIndex % g_width) / cast(float)g_width * 2.0f - 1.0f + uniform(-pixelSizeX, pixelSizeX, gen);
  float y = cast(float)(pixelIndex / g_width) / cast(float)g_height * 2.0f - 1.0f + uniform(-pixelSizeY, pixelSizeY, gen);
  return g_camera.getScreenRay(x, y);
}

float haltonSequence(float index, float base)
{
  float result = 0;
  float f = 1.0f / base;
  while(index > 0.0f)
  {
    result = result + f * (core.stdc.math.fmod(index, base));
    index = floorf(index / base);
    f = f / base;
  }
  return result;
}

/*
psi = 0..360
phi = 0..90

(cos(psi) -sin(psi)   0)     (cos(phi))     (cos(psi) * cos(phi))
(sin(psi)  cos(psi)   0)  *  (0       )  =  (sin(psi) * cos(phi))
(       0         0   1)     (sin(phi))  =  (sin(phi)           )
*/

vec2 ConcentricSampleDisk(vec2 uv)
{
  uv = uv * 2.0f - vec2(1.0f);
  float r, theta;
  if(uv.x == 0.0f && uv.y == 0.0f)
    return vec2(0.0f);

  if(uv.x >= -uv.y) 
  {
    if(uv.x > uv.y)
    {
      r = uv.x;
      theta = (uv.y > 0.0f) ? theta = uv.y / r : 8.0f + uv.y / r;
    }
    else
    {
      r = uv.y;
      theta = 2.0f - uv.x / r;
    }
  }
  else
  {
    if(uv.x <= uv.y)
    {
      r = -uv.x;
      theta = 4.0f - uv.y / r;
    }
    else
    {
      r = -uv.y;
      theta = 6.0f + uv.x / r;
    }
  }
  theta *= PI / 4.0f;
  return vec2(r * cosf(theta), r * sinf(theta));
}

vec3 UniformSampleHemisphere(vec2 uv)
{
	float z = 1.0f - 2.0f * uv.x;
	float r = sqrtf(max(0.0f, 1.0f - z*z));
	float phi = 2.0f * PI * uv.y;
	float x = r * cosf(phi);
	float y = r * sinf(phi);
	return vec3(x, y, z);
}

vec3 CosineSampleHemisphere(vec2 uv)
{
  auto result = vec3(ConcentricSampleDisk(uv), 0.0f);
  result.z = sqrtf(max(0.0f, 1.0f - result.x * result.x - result.y * result.y));
  return result;
}

vec3 angleToLocalDirection(float phi, float psi)
{
  float cosPhi = cosf(phi);
  auto result = vec3(cosf(psi) * cosPhi, sinf(psi) * cosPhi, sinf(phi));
  return result;
}

vec3 toWorldSpace(vec3 localDir, ref const(vec3) normal)
{
  auto up = normal;
  auto dir = vec3(1,0,0);
  if(abs(dir.dot(up)) > 0.9f)
  {
    dir = vec3(0,1,0);
  }
  auto right = up.cross(dir).normalized();
  dir = up.cross(right).normalized();
  return mat3(dir, right, up) * localDir;
}

vec3 sampleSky(vec3 dir)
{
  auto sunDir = vec3(1, 1, 6).normalized();
  float dot = sunDir.dot(dir);
  if(dot > 0.997f)
    return vec3(1.0f, 0.984f, 0.8f) * 30.0f;
  return vec3(0.682f, 0.977f, 1.0f);
}

/+void computeL(uint pixelIndex, ref Pixel pixel, ref Random gen)
{
  Ray viewRay = getViewRay(pixelIndex);
  float rayPos = 0.0f;
  vec3 normal;
  const(Scene.TriangleData)* data;
  if( g_scene.trace(viewRay, rayPos, normal, data)){
    if(data.material.emissive > 0.0f)
    {
      pixel.color.x = pixel.color.y = pixel.color.z = 1.0f;
    }
    else
    {
      float NdotL = abs(normal.dot(-viewRay.dir));
      pixel.color = data.material.color * NdotL;
    }
    /*float NdotL = abs(normal.dot(-viewRay.dir));
    pixel.color.x = NdotL;
    pixel.color.y = NdotL;
    pixel.color.z = NdotL;*/
  }
  else
  {
    pixel.color = vec3(0,0,0);
  }
}+/

enum float BRDF = 1.0f / PI;

/*vec3 computeLrefl(vec3 pos, vec3 theta, ref const(vec3) normal, const(Scene.TriangleData)* data, ref Random gen, uint depth)
{
  return computeLdirect(pos, theta, normal, data, gen) +
         computeLindirect(pos, theta, normal, data, gen, depth);
}

vec3 computeLindirect(vec3 pos, vec3 theta, ref const(vec3) normal, const(Scene.TriangleData)* data, ref Random gen, uint depth)
{
  float a = uniform(0.0f, 1.0f, gen);
  float absorb = 0.7f;
  if(a > absorb || depth > 5)
    return vec3(0.0f, 0.0f, 0.0f);
	float psi = uniform(0, 2 * PI, gen);
	float phi = uniform(0, PI_2, gen);
	vec3 sampleDir = angleToDirection(phi, psi, normal);
	Ray sampleRay = Ray(pos + normal * 0.1f, sampleDir);
	float hitDistance = 0.0f;
	vec3 hitNormal;
	const(Scene.TriangleData)* hitData;
	if( g_scene.trace(sampleRay, hitDistance, hitNormal, hitData)){
		vec3 hitPos = sampleRay.get(hitDistance);
		return (BRDF * PI * data.material.color * computeLrefl(hitPos, -sampleDir, hitNormal, hitData, gen, depth + 1));
	}
	return vec3(0.0f, 0.0f, 0.0f);
}

vec3 computeLdirect(vec3 x, vec3 theta, ref const(vec3) normalX, const(Scene.TriangleData)* data, 
                    ref Random gen)
{
  vec3 L;
  if(g_hasLights)
  {
    vec3 y = pickRandomLightPoint(gen);
    float lightDistance = (y - x).length;
    vec3 phi = (y - x).normalized();

    auto shadowRay = Ray(x + normalX * FloatEpsilon, phi); 

    // compute V(x, y)
	  float distanceY = 0.0f;
	  vec3 normalY;
	  const(Scene.TriangleData)* dataY;
    if(normalX.dot(phi) > FloatEpsilon && g_scene.trace(shadowRay, distanceY, normalY, dataY))
    {
      if(distanceY < lightDistance - FloatEpsilon)
      {
        return vec3(0.0f, 0.0f, 0.0f);
      }
      vec3 Le = dataY.material.emissive * dataY.material.color;
      float G = normalX.dot(phi) * normalY.dot(-phi) / (lightDistance * lightDistance);
      L = (BRDF * data.material.color ) * Le * G * g_totalLightSourceArea;
    }
  }
  if(g_hasSky)
  {
    //first compute contribution of sun
    {
      float psi = uniform(0, 2 * PI, gen);
      float phi = uniform(PI_2 - PI_2 * 0.01f, PI_2, gen);
      vec3 sampleDir = angleToDirection(phi, psi, g_sunDir);
      if(normalX.dot(sampleDir) > FloatEpsilon)
      {

        auto sunRay = Ray(x + normalX * 0.1f, sampleDir);
        if(g_scene.hitsNothing(sunRay))
        {
          L += (BRDF * data.material.color) * g_sunRadiance * normalX.dot(g_sunDir);
        }
      }
    }

    // second compute contribution of sky
    {
      float psi = uniform(0, 2 * PI, gen);
      float phi = uniform(0, PI_2, gen);
      vec3 sampleDir = angleToLocalDirection(phi, psi);
      if(normalX.dot(sampleDir) > FloatEpsilon)
      {
        Ray sampleRay = Ray(x + normalX * 0.1f, sampleDir);
        if(g_scene.hitsNothing(sampleRay)){
          L += (BRDF * data.material.color) * g_skyRadiance * normalX.dot(sampleDir);
        }
      }
    }
  }
  return L;
}

void computeL(uint pixelIndex, ref Pixel pixel, ref Random gen)
{
	enum uint N = 5;
	Ray viewRay = getViewRay(pixelIndex, gen);
	float rayPos = 0.0f;
	vec3 normal;
	const(Scene.TriangleData)* data;
	if( g_scene.trace(viewRay, rayPos, normal, data)){
    vec3 hitPos = viewRay.get(rayPos);
    vec3 sum;
    for(uint i=0; i<N; i++){
			sum += computeLindirect(hitPos, -viewRay.dir, normal, data, gen, 0);
    }
    pixel.sum += data.material.emissive * data.material.color + computeLdirect(hitPos, -viewRay.dir, normal, data, gen) + sum / cast(float)N;
	}
	else{
		pixel.sum += sampleSky(viewRay.dir);
	}
	pixel.n += 1;
	pixel.color = pixel.sum / pixel.n;
}*/

/**
* computes the output color of the generated image
* 
* Params: 
*  pixelOffset = the pixel offset (for multithreading)
*  pixels = the pixels that should be computed
*  gen = the random number generator
*/
/*void computeOutputColor(uint pixelOffset, Pixel[] pixels, ref Random gen)
{
  foreach(uint pixelIndex, ref pixel; pixels)
  {
    computeL(pixelOffset + pixelIndex, pixel, gen);
  }                          
}*/