module main;

import std.random;
import thBase.io;
import thBase.math;
import thBase.asserthandler;
import thBase.timer;
import thBase.task;
import thBase.file;
import thBase.math3d.all;
import thBase.algorithm;
import thBase.dds;
import thBase.format;
import thBase.asserthandler;
import core.thread;
static import core.cpuid;

import std.math;
import core.stdc.math : cosf, sinf;
import core.stdc.stdlib;

import sdl;
import rendering;
import scene;

enum uint numSamples = 32;
enum uint additionalSkySamples = 512;
enum uint directLightSamples = 128;
enum bool extrapolateGeometryEnabled = true;
enum bool useBlackAmbient = true;
enum bool useCosineDistribution = true;
enum bool computeDirectLightEnabled = true;

__gshared vec2 g_blackPixelLocation = vec2(1.0f, 0.0f);


enum uint numPrecomputedSamples = 1024;
__gshared vec2[numPrecomputedSamples][128] g_precomputedSamples;

//version = PerformanceTest;

uint g_workerId = 0;
struct PerWorkerData
{
  uint wastedSamples = 0;
  uint backfaceSamples = 0;
  uint totalSamples = 0;
  byte[256-8] padding;
}
PerWorkerData[16] g_perWorkerData;

void setPixel(SDL.Surface *screen, int x, int y, ubyte r, ubyte g, ubyte b)
{
  if(x < 0 || x >= g_width || y < 0 || y >= g_height)
    return;
  uint *pixmem32;
  uint colour;  

  colour = SDL.MapRGB( screen.format, r, g, b );

  pixmem32 = cast(uint*)(screen.pixels + (y * screen.pitch + x * 4));
  *pixmem32 = colour;
}

void drawLine( SDL.Surface *screen, vec2 p1, vec2 p2 )
{
  float x1 = p1.x;
  float x2 = p2.x;
  float y1 = p1.y;
  float y2 = p2.y;
  // Bresenham's line algorithm
  const bool steep = (fabs(y2 - y1) > fabs(x2 - x1));
  if(steep)
  {
    swap(x1, y1);
    swap(x2, y2);
  }

  if(x1 > x2)
  {
    swap(x1, x2);
    swap(y1, y2);
  }

  const float dx = x2 - x1;
  const float dy = fabs(y2 - y1);

  float error = dx / 2.0f;
  const int ystep = (y1 < y2) ? 1 : -1;
  int y = cast(int)y1;

  const int maxX = cast(int)x2;

  for(int x=cast(int)x1; x<maxX; x++)
  {
    if(steep)
    {
      setPixel(screen, y, x, 255, 0, 0);
    }
    else
    {
      setPixel(screen, x, y, 255, 0, 0);
    }

    error -= dy;
    if(error < 0)
    {
      y += ystep;
      error += dx;
    }
  }
}

void drawScreen(SDL.Surface* screen, Pixel[] pixels)
{ 
  immutable(float) a = 0.055f;
  immutable(float) aPlusOne = 1 + 0.055f;
  immutable(float) power = 1 / 2.4f;
  /*version(USE_SSE)
  {
    float one = 1.0f;
    float zero = 0.0f;
    float scale = 255.0f;
    asm {
      movss XMM0, one;
      pshufd XMM0, XMM0, 0b00_00_00_00; //shuffle xxxx
      movss XMM1, zero;
      pshufd XMM1, XMM1, 0b00_00_00_00; //shuffle xxxx
      movss XMM2, scale;
      pshufd XMM2, XMM2, 0b00_00_00_00; //shuffle xxxx
    }
    for(int y = 0; y < screen.height; y++ ) 
    {
      for(int x = 0; x < screen.width; x++ ) 
      {
        Pixel* p = &pixels[g_width * y + x];
        uint[4] colors;
        float[4] linear;
        asm {
          mov EAX, p;
          movups XMM3, [EAX];
          minps XMM3, XMM0; //min(x, 1)
          maxps XMM3, XMM1; //max(x, 0)
          lea EAX, linear;
          movups [EAX], XMM3;
        }
        linear[0] = (linear[0] <= 0.0031308f) ? linear[0] * 12.92f : aPlusOne * powf(linear[0], power) - a;
        linear[1] = (linear[1] <= 0.0031308f) ? linear[1] * 12.92f : aPlusOne * powf(linear[1], power) - a;
        linear[2] = (linear[2] <= 0.0031308f) ? linear[2] * 12.92f : aPlusOne * powf(linear[2], power) - a;

        asm {
          lea EAX, linear;
          movups XMM3, [EAX];
          mulps XMM3, XMM2; //x *= 255.0f
          cvtps2dq XMM3, XMM3; //convert to int
          lea EAX, colors;
          movups [EAX], XMM3;
        }
        setPixel(screen, x, y, cast(ubyte)colors[0], cast(ubyte)colors[1], cast(ubyte)colors[2]);
      }
    }
  }
  else*/
  {
    for(int y = 0; y < screen.height; y++ ) 
    {
      for(int x = 0; x < screen.width; x++ ) 
      {
        Pixel* p = &pixels[g_width * y + x];
        float r = saturate(p.color.x);
        float g = saturate(p.color.y);
        float b = saturate(p.color.z);
        //r = (r <= 0.0031308f) ? r * 12.92f : aPlusOne * powf(r, power) - a;
        //g = (g <= 0.0031308f) ? g * 12.92f : aPlusOne * powf(g, power) - a;
        //b = (b <= 0.0031308f) ? b * 12.92f : aPlusOne * powf(b, power) - a;
        setPixel(screen, x, y, cast(ubyte)(r * 255.0f), cast(ubyte)(g * 255.0f), cast(ubyte)(b * 255.0f));
      }
    }
  }

  SDL.Flip(screen); 
}

__gshared bool g_run = true;

// Task which triggeres the computation
class PerPixelTask : Task
{
  private:
    uint m_pixelOffset;
    Pixel[] m_pixels;
    Random m_gen;
    void function(uint offset, Pixel[], ref Random) m_func;

  public:
    this(TaskIdentifier identifier, void function(uint offset, Pixel[], ref Random) func, uint pixelOffset, Pixel[] pixels)
    {
      super(identifier);
      m_pixelOffset = pixelOffset;
      m_pixels = pixels;
      m_gen.seed(m_pixelOffset + cast(uint)pixels[0].n);
      m_func = func;
    }

    override void Execute()
    {
      //computeOutputColor(m_pixelOffset, m_pixels, m_gen);
      m_func(m_pixelOffset, m_pixels, m_gen);
    }

    override void OnTaskFinished() {}
}

// Worker thread
class Worker : Thread
{
  uint m_id;
  this(uint id)
  {
    m_id = id;
    super(&run);
  }

  void run()
  {
    g_workerId = m_id;
    g_localTaskQueue.executeTasksUntil( (){ return !g_run; } );
  }
}

auto interpolate(T)(float u, float v, T val0, T val1, T val2)
{
  if(u < 0.0f)
    u = 0.0f;
  if(v < 0.0f)
    v = 0.0f;
  float uv = (u + v);
  float u1, v1;
  if(uv != 0.0f)
  {
    float x = 1.0f / uv;
    u1 = x * u;
    v1 = x * v;
  }
  else
  {
    u1 = u;
    v1 = v;
  }
  immutable float sqrt2 = 1.414213562f;
  float d1 = sqrt((1.0f-u1)*(1.0f-u1) + v1*v1) / sqrt2;
  float d2 = sqrt(u1*u1 + (1.0f-v1)*(1.0f-v1)) / sqrt2;
  auto interpolated1 = val1 * d1 + val2 * d2;

  float i1;
  if(uv != 0.0f)
  {
    float len = sqrt(u1*u1 + v1*v1);
    i1 = sqrt(u*u+v*v) / len; 
  }
  else
  {
    i1 = 0.0f;
  }
  float i2 = 1.0f - i1;

  return val0 * i2 + interpolated1 * i1;
}

auto extrapolate(T)(float u, float v, T val0, T val1, T val2)
{
  float uv = (u + v);
  float u1, v1;
  if(uv != 0.0f)
  {
    float x = 1.0f / uv;
    u1 = x * u;
    v1 = x * v;
  }
  else
  {
    u1 = u;
    v1 = v;
  }
  immutable float sqrt2 = 1.414213562f;
  float d1 = sqrt((1.0f-u1)*(1.0f-u1) + v1*v1) / sqrt2;
  float d2 = sqrt(u1*u1 + (1.0f-v1)*(1.0f-v1)) / sqrt2;
  auto interpolated1 = val1 * d1 + val2 * d2;

  float i1;
  if(uv != 0.0f)
  {
    float len = sqrt(u1*u1 + v1*v1);
    i1 = sqrt(u*u+v*v) / len; 
  }
  else
  {
    i1 = 0.0f;
  }
  float i2 = 1.0f - i1;

  return val0 * i2 + interpolated1 * i1;
}

vec2 computeUV(vec2 val, vec2 v0, vec2 v1, vec2 v2)
{
  auto p = v0;
  auto r1 = (v1 - v0);
  auto r2 = (v2 - v0);
  float d = (r1.x * r2.y - r1.y * r2.x);
  if(d.epsilonCompare(0.0f))
  {
    return vec2(float.max, float.max);
  }
  float u = (-p.x * r2.y + p.y * r2.x - r2.x * val.y + r2.y * val.x) / d;
  float v = ( p.x * r1.y - p.y * r1.x + r1.x * val.y - r1.y * val.x) / d;
  return vec2(u, v);
}

float g_cylinderRadius = 0.50f;

vec3 mapToCylinder(vec2 p)
{
  return vec3(cosf(p.x * PI * 2.0) * g_cylinderRadius, sinf(p.x * PI * 2.0) * g_cylinderRadius, p.y);
}

vec3 mapToHemisphere(vec2 p)
{
  float phi = p.y * PI / 2.0; // 0..90
  float psi = p.x * PI * 2.0; // 0..360
  float cosPhi = cosf(phi);
  auto result = vec3(cosf(psi) * cosPhi, sinf(psi) * cosPhi, sinf(phi));
  return result;
}

float minDist2D(vec2 p, vec2[] other)
{
  float dist = (other[0] - p).squaredLength;
  foreach(cur; other[1..$])
  {
    float dist2 = (cur - p).squaredLength;
    if(dist2 < dist)
      dist = dist2;
  }
  return dist;
}

double minDistCylinder(vec2 p, vec2[] other)
{
  float dist = (other[0].mapToCylinder - p.mapToCylinder).squaredLength;
  foreach(cur; other[1..$])
  {
    float dist2 = (cur.mapToCylinder - p.mapToCylinder).squaredLength;
    if(dist2 < dist)
      dist = dist2;
  }
  return dist;
}

float minDistHemisphere(vec2 p, vec2[] other)
{
  float dist = (other[0].mapToHemisphere - p.mapToHemisphere).squaredLength;
  foreach(cur; other[1..$])
  {
    float dist2 = (cur.mapToHemisphere - p.mapToHemisphere).squaredLength;
    if(dist2 < dist)
      dist = dist2;
  }
  return dist;
}

void bestCanidatePattern(alias distanceFunc)(vec2[] pattern, ref Random gen)
{
  size_t numValid = 1;
  pattern[0].x = uniform(0.0f, 1.0f, gen);
  pattern[0].y = uniform(0.0f, 1.0f, gen);
  foreach(ref p; pattern[1..$])
  {
    vec2 best;
    best.x = uniform(0.0f, 1.0f, gen);
    best.y = uniform(0.0f, 1.0f, gen);
    float dist = distanceFunc(best, pattern[0..numValid]);
    for(uint i=0; i < 2000; i++)
    {
      vec2 cur;
      cur.x = uniform(0.0f, 1.0f, gen);
      cur.y = uniform(0.0f, 1.0f, gen);
      float dist2 = distanceFunc(cur, pattern[0..numValid]);
      if(dist2 > dist)
      {
        dist = dist2;
        best = cur;
      }
    }
    p = best;
    numValid++;
  }
}

void precomputeSamples(ref Random gen)
{
  char[1024] buffer;
  auto len = formatStatic(buffer, "samples%s.dat", numPrecomputedSamples); 
  //auto len = formatStatic(buffer, "samples.dat"); 
  {
    auto samplesFile = RawFile(buffer[0..len], "rb");
    if(samplesFile.isOpen && samplesFile.size == g_precomputedSamples.sizeof)
    {
      samplesFile.readArray(g_precomputedSamples[]);
      return;
    }
  }

  foreach(size_t i, ref samples; g_precomputedSamples)
  {
    writefln("pattern %d of %d", i, g_precomputedSamples.length);
    bestCanidatePattern!minDist2D(samples, gen);
    /*static if(useCosineDistribution)
      bestCanidatePattern!minDistCylinder(samples, gen);
    else
      bestCanidatePattern!minDistHemisphere(samples, gen);*/
  }
  auto samplesFile = RawFile(buffer[0..len], "wb");
  samplesFile.writeArray(g_precomputedSamples[]);
}

void pickPrecomputedSample(vec2[] pattern, ref Random gen)
{
  uint index = uniform(0, g_precomputedSamples.length, gen);
  pattern[] = g_precomputedSamples[index];
}

struct Edge
{
  uint minY, maxY;
  float xs; // intersection with the current scanline
  float invM;
  vec2 uvDelta;
  vec2 uv;

  this(float x1, float y1, vec2 uv1, float x2, float y2, vec2 uv2)
  {
    uvDelta = (uv2 - uv1) / (y2 - y1);
    x1 = floor(x1);
    y1 = floor(y1);
    x2 = floor(x2);
    y2 = floor(y2);
    assert(!epsilonCompare(y1, y2)); //TODO fix properly

    if(y1 < y2)
    {
      minY = cast(uint)y1;
      maxY = cast(uint)y2;
      uv = uv1;
    }
    else
    {
      minY = cast(uint)y2;
      maxY = cast(uint)y1;
      uv = uv2;
    }

    float deltaX = (x2 - x1);
    if(epsilonCompare(deltaX, 0.0f))
    {
      xs = x1;
      invM = 0.0f;
    }
    else
    {
      auto m = (y2 - y1) / deltaX;
      invM = 1.0f / m;
      float b = y1 - (m * x1);
      xs = round((minY - b) * invM);
    }
  }
}

void rasterTriangles(size_t from, size_t to, Pixel[] pixels)
{
  float fHeight = cast(float)g_height;
  float fWidth = cast(float)g_width;

  foreach(size_t i, ref t; g_scene.triangleData[from..to])
  {
    auto triangle = &g_scene.triangles[i + from];
    vec2[3] verts;
    verts[] = t.tex[];
    verts[0] *= vec2(fWidth, fHeight);
    verts[1] *= vec2(fWidth, fHeight);
    verts[2] *= vec2(fWidth, fHeight);

    vec2[3] uvs;
    uvs[0] = vec2(0.0f, 0.0f);
    uvs[1] = vec2(0.0f, 1.0f);
    uvs[2] = vec2(1.0f, 0.0f);

    vec3[3] wsPos = [ triangle.v0, triangle.v1, triangle.v2 ];

    if(verts[0].y > verts[1].y)
    {
      swap(verts[0], verts[1]);
      swap(uvs[0], uvs[1]);
    }
    if(verts[0].y > verts[2].y)
    {
      swap(verts[1], verts[2]);
      swap(verts[0], verts[1]);
      swap(uvs[1], uvs[2]);
      swap(uvs[0], uvs[1]);
    }
    else if(verts[1].y > verts[2].y)
    {
      swap(verts[1], verts[2]);
      swap(uvs[1], uvs[2]);
    }

    verts[0] = floor(verts[0]);
    verts[1] = floor(verts[1]);
    verts[2] = floor(verts[2]);
    if(verts[0].x == verts[1].x && verts[1].x == verts[2].x)
      continue;
    if(verts[0].y == verts[1].y && verts[1].y == verts[2].y)
      continue;

    Edge[3] edges;
    uint numEdges = 0;

    if(cast(uint)(verts[0].y) != cast(uint)(verts[1].y))
    {
      edges[numEdges++] = Edge(verts[0].x, verts[0].y, uvs[0], verts[1].x, verts[1].y, uvs[1]);
      edges[numEdges++] = Edge(verts[0].x, verts[0].y, uvs[0], verts[2].x, verts[2].y, uvs[2]);
      if(cast(uint)(verts[1].y) != cast(uint)(verts[2].y))
      {
        edges[numEdges++] = Edge(verts[1].x, verts[1].y, uvs[1], verts[2].x, verts[2].y, uvs[2]);
      }
    }
    else
    {
      edges[numEdges++] = Edge(verts[0].x, verts[0].y, uvs[0], verts[2].x, verts[2].y, uvs[2]);
      edges[numEdges++] = Edge(verts[1].x, verts[1].y, uvs[1], verts[2].x, verts[2].y, uvs[2]);
    }

    if(edges[0].xs > edges[1].xs || (epsilonCompare(edges[0].xs,edges[1].xs) && edges[0].invM > edges[1].invM))
      swap(edges[0], edges[1]);
    for(uint y = edges[0].minY; true; y++)
    {
      if(edges[0].maxY <= y)
      {
        if(numEdges < 3)
          break;
        numEdges--;
        edges[2].xs += edges[2].invM * (edges[2].minY - y);
        swap(edges[0], edges[2]);
        if(edges[0].xs > edges[1].xs || (epsilonCompare(edges[0].xs,edges[1].xs) && edges[0].invM > edges[1].invM))
          swap(edges[0], edges[1]);
      }
      else if(edges[1].maxY <= y)
      {
        if(numEdges < 3)
          break;
        numEdges--;
        edges[2].xs += edges[2].invM * (edges[2].minY - y);
        swap(edges[1], edges[2]);
        if(edges[0].xs > edges[1].xs || (epsilonCompare(edges[0].xs,edges[1].xs) && edges[0].invM > edges[1].invM))
          swap(edges[0], edges[1]);
      }
      //assert(edges[0].xs <= edges[1].xs);
      uint end = cast(uint)edges[1].xs;
      if(end >= g_width)
        end = g_width;
      uint start = cast(uint)(edges[0].xs < 0.0f ? 0.0f : edges[0].xs);
      vec2 uv = edges[0].uv;
      vec2 uvDelta = (edges[1].uv - edges[0].uv) / cast(float)(end - start + 1);
      for(uint x = start; x <= end; x++)
      {
        uv += uvDelta;
        auto curPixel = &pixels[y * g_width + x];
        curPixel.rastered = true;
        curPixel.position = interpolate(uv.x, uv.y, wsPos[0], wsPos[1], wsPos[2]);
        curPixel.normal = interpolate(uv.x, uv.y, t.n[0], t.n[1], t.n[2]).normalized;
        curPixel.directLight = t.material.color * t.material.emissive;
        vec2 coords = interpolate(uv.x, uv.y, t.tex[0], t.tex[1], t.tex[2]);
        //curPixel.color = curPixel.position * (1.0f / 20.0f) + vec3(0.5f);
        //curPixel.color = vec3(coords.x, coords.y, 0.0f);
        //vec2 uv2 = uv * 0.5f + vec2(0.5f);
        //curPixel.color = vec3(uv2.x, uv2.y, uv2.x < 0.0f || uv2.y < 0.0f ? 1.0f : 0.0f);
        curPixel.color = curPixel.normal * 0.5f + vec3(0.5f);
        //curPixel.color = vec3(1.0f, 0.0f, 0.0f);
        //curPixel.color = t.n[0] * 0.5f + vec3(0.5f);
        //curPixel.color = vec3(cast(float)i / cast(float)g_scene.triangles.length);
        
      }
      edges[0].xs += edges[0].invM;
      edges[0].uv += edges[0].uvDelta;
      edges[1].xs += edges[1].invM;
      edges[1].uv += edges[1].uvDelta;
    }
  }
}

void extrapolateGeometry(uint offset, Pixel[] pixels, ref Random gen)
{
  auto edges = g_scene.textureEdges.query(Rectangle(vec2(0, 0), vec2(0.01, 0.01)));
  auto pixelSize = vec2(1.0f / g_width, 1.0f / g_height);
  auto searchDistance = pixelSize * 3.0f;
  float maxDist = 2.0f / g_width;
  vec2[3] verts;

  foreach(size_t i, ref pixel; pixels)
  {
    if(!pixel.rastered)
    {
      uint x = (offset + i) % g_width;
      uint y = (offset + i) / g_width;
      vec2 pos = vec2(cast(float)x, cast(float)y) * pixelSize + pixelSize * 0.5f;
      edges = g_scene.textureEdges.query(Rectangle(pos - searchDistance, pos + searchDistance), move(edges));
      if(!edges.empty)
      {
        auto closestEdge = edges.front;
        auto closestDistance = closestEdge.distance(pos);
        edges.popFront();
        foreach(edge; edges)
        {
          float dist = edge.distance(pos);
          if(dist < closestDistance)
          {
            closestDistance = dist;
            closestEdge = edge;
          }
        }
        if(closestDistance <= maxDist)
        {
          auto triangleData = g_scene.triangleData[closestEdge.triangleIndex];
          auto triangle = g_scene.triangles[closestEdge.triangleIndex];

          auto uv = computeUV(pos, triangleData.tex[0], triangleData.tex[2], triangleData.tex[1]);
          pixel.position = extrapolate(uv.x, uv.y, triangle.v0, triangle.v1, triangle.v2);
          pixel.normal = extrapolate(uv.x, uv.y, triangleData.n[0], triangleData.n[1], triangleData.n[2]).normalized;
          pixel.color = pixel.normal * 0.5f + vec3(0.5f);
          pixel.rastered = true;
          //pixel.color = vec3(closestDistance / maxDist, 1, 1);
          //vec2 uv2 = uv * 0.5f + vec2(0.5f);
          //pixel.color = vec3(uv2.x, uv2.y, 0.0f);
          //pixel.color = pixel.position * (1.0f / 20.0f) + vec3(0.5f);
          //pixel.color = vec3(cast(float)closestEdge.triangleIndex / cast(float)g_scene.triangles.length);
        }
      }
    }
  }
}

void takeSamples(uint offset, Pixel[] pixels, ref Random gen)
{
  vec2[] pattern = (cast(vec2*)alloca(vec2.sizeof * numPrecomputedSamples))[0..numPrecomputedSamples];
  PerWorkerData* perWorkerData = &g_perWorkerData[g_workerId];
  foreach(ref pixel; pixels)
  {
    if(!pixel.rastered)
      continue;
    //bestCanidatePattern!(minDistCylinder)(pattern, gen);
    pickPrecomputedSample(pattern, gen);
    size_t i = 0;
    foreach(ref sample; pixel.samples)
    {
      start:
      i++;
      static if(useCosineDistribution)
	    vec3 sampleDir = toWorldSpace(CosineSampleHemisphere(pattern[i]), pixel.normal);
	  else
		vec3 sampleDir = toWorldSpace(UniformSampleHemisphere(pattern[i]), pixel.normal);
	    Ray sampleRay = Ray(pixel.position + pixel.normal * 0.1f, sampleDir);
	    float hitDistance = 0.0f;
	    vec2 hitTexcoords;
	    const(Scene.TriangleData)* hitData;
      auto traceResult = g_scene.trace(sampleRay, hitDistance, hitTexcoords, hitData, IgnoreBackfaces.no);
      perWorkerData.totalSamples++;
	    if( traceResult == TraceResult.FrontFaceHit )
      {
		    sample = hitTexcoords;
	    }
      else if(traceResult == TraceResult.BackFaceHit )
      {
        sample = g_blackPixelLocation;
        perWorkerData.backfaceSamples++;
      }
      else
      {
        // nothing hit
        if(i < numPrecomputedSamples - numSamples)
        {
          pixel.numSkippedSamples++;
          goto start;
        }
        else
        {
          sample = g_blackPixelLocation;
          perWorkerData.wastedSamples++;
        }
      }
      /*if(i==1)
      {
        pixel.color = vec3(hitTexcoords.x, hitTexcoords.y, 0.0f);
      }*/
    }
    static if(!useBlackAmbient)
    {
      for(; i < numPrecomputedSamples; i++)
      {
	      vec3 sampleDir = toWorldSpace(CosineSampleHemisphere(pattern[i]), pixel.normal);
        if(sampleDir.z > 0.0f)
        {
	        Ray sampleRay = Ray(pixel.position + pixel.normal * 0.1f, sampleDir);
	        if( g_scene.hitsNothing(sampleRay) )
          {
		        pixel.ambient++;
	        }
        }
      }
      for(uint j=0; j < additionalSkySamples; j++)
      {
        vec3 sampleDir = toWorldSpace(CosineSampleHemisphere(vec2(uniform(0.0f, 1.0f, gen), uniform(0.0f, 1.0f, gen))), pixel.normal);
        if(sampleDir.z > 0.0f)
        {
          Ray sampleRay = Ray(pixel.position + pixel.normal * 0.1f, sampleDir);
          float hitDistance = 0.0f;
          vec2 hitTexcoords;
          const(Scene.TriangleData)* hitData;
          if( g_scene.hitsNothing(sampleRay) )
          {
            pixel.ambient++;
          }
        }
      }
    }

    pixel.color = vec3((cast(float)pixel.ambient / cast(float)(numPrecomputedSamples + additionalSkySamples)), 
                       0.0f, //(cast(float)pixel.numSkyRays / cast(float)(numSamples*2)), 
                       0.0f);
  }
}

vec3 computeLDirect(vec3 x, vec3 normalX, ref Random gen)
{
  vec3 L;
  for(uint lightIndex=0; lightIndex<g_lightTriangles.length; lightIndex++)
  {
    vec3 y = pickRandomLightPoint(gen, lightIndex);
    float lightDistance = (y - x).length;
    vec3 phi = (y - x).normalized();

    auto shadowRay = Ray(x + normalX * FloatEpsilon, phi); 

    // compute V(x, y)
    float distanceY = 0.0f;
    vec3 normalY;
    const(Scene.TriangleData)* dataY;
    if(normalX.dot(phi) > FloatEpsilon && g_scene.trace(shadowRay, distanceY, normalY, dataY))
    {
      if(distanceY > lightDistance - FloatEpsilon)
      {
        vec3 Le = dataY.material.emissive * dataY.material.color;
        float G = normalX.dot(phi) * normalY.dot(-phi) / (lightDistance * lightDistance);
        if(!(G > 0.0f)) G = 0.0f;
        assert(G == G);
        assert(G >= 0.0f);
        L += BRDF * Le * G * g_totalLightSourceArea[lightIndex];
      }
    }
  }
  return L;
}

void computeDirectLight(uint offset, Pixel[] pixels, ref Random gen)
{
  foreach(ref pixel; pixels)
  {
    if(!pixel.rastered)
      continue;
    vec3 sum = vec3(0.0f);
    for(uint i=0; i < 16; i++)
    {
      sum += computeLDirect(pixel.position, pixel.normal, gen); 
    }
    if(sum.squaredLength > FloatEpsilon)
    {
      for(uint i=16; i < directLightSamples; i++)
      {
        sum += computeLDirect(pixel.position, pixel.normal, gen); 
      }
    }
    sum = sum / cast(float)directLightSamples;
    pixel.directLight += sum;
    pixel.color = pixel.directLight;
  }
}

void writeDDSFiles(uint width, uint height, Pixel[] pixels)
{
  {
    auto data = NewArray!ushort(width * height * 4);
    scope(exit) Delete(data);
    uint numFiles = (Pixel.samples.length + 1) / 2;
    for(uint i=0; i<numFiles; i++)
    {
      for(uint y=0; y<height; y++)
      {
        for(uint x=0; x<width; x++)
        {
          data[y * width * 4 + x * 4] = cast(ushort)(pixels[y * width + x].samples[i*2].x * 65535.0f);
          data[y * width * 4 + x * 4 + 1] = cast(ushort)(pixels[y * width + x].samples[i*2].y * 65535.0f);
          data[y * width * 4 + x * 4 + 2] = cast(ushort)(pixels[y * width + x].samples[i*2+1].x * 65535.0f);
          data[y * width * 4 + x * 4 + 3] = cast(ushort)(pixels[y * width + x].samples[i*2+1].y * 65535.0f);
        }
      }
      char[256] name;
      auto len = formatStatic(name, "gi%d.dds", i);
      WriteDDS(name[0..len], width, height, DDSLoader.DXGI_FORMAT.R16G16B16A16_UNORM, (cast(void*)data.ptr)[0..(width * height * 4 * ushort.sizeof)]);
      writefln("%s written", name[0..len]);
    }
  }

  uint maxSkippedSamples = 0;
  {
    auto data = NewArray!ushort(width * height * 2);
    scope(exit) Delete(data);
    for(uint y=0; y<height; y++)
    {
      for(uint x=0; x<width; x++)
      {
        static if(useBlackAmbient)
          float ambient = 0.0f;
        else
          float ambient = cast(float)pixels[y * width + x].ambient / cast(float)(numPrecomputedSamples + additionalSkySamples);
        data[y * width * 2 + x * 2] = cast(ushort)(ambient * 65535.0f);
        data[y * width * 2 + x * 2 + 1] = cast(ushort)(pixels[y * width + x].numSkippedSamples);
        if(pixels[y * width + x].rastered)
          maxSkippedSamples = max(maxSkippedSamples, pixels[y * width + x].numSkippedSamples);
      }
    }
    WriteDDS("sky.dds", width, height, DDSLoader.DXGI_FORMAT.R16G16_UNORM, (cast(void*)data.ptr)[0..(width * height * 2 * ushort.sizeof)]);
    writefln("sky.dds written");
  }
  writefln("maxSkippedSamples = %d", maxSkippedSamples);

  {
    auto data = NewArray!float(width * height * 4);
    scope(exit) Delete(data);
    for(uint y=0; y<height; y++)
    {
      for(uint x=0; x<width; x++)
      {
        data[y * width * 4 + x * 4] = pixels[y * width + x].position.x;
        data[y * width * 4 + x * 4 + 1] = pixels[y * width + x].position.y;
        data[y * width * 4 + x * 4 + 2] = pixels[y * width + x].position.z;
        data[y * width * 4 + x * 4 + 3] = 0.0f;
      }
    }
    WriteDDS("position.dds", width, height, DDSLoader.DXGI_FORMAT.R32G32B32A32_FLOAT, (cast(void*)data.ptr)[0..(width * height * 4 * float.sizeof)]);
    writefln("position.dds written");
  }

  {
    auto data = NewArray!ubyte(width * height * 4);
    scope(exit) Delete(data);
    for(uint y=0; y<height; y++)
    {
      for(uint x=0; x<width; x++)
      {
        vec3 normal = pixels[y * width + x].normal * 0.5f + vec3(0.5f);
        data[y * width * 4 + x * 4] = cast(ubyte)(normal.x * 255.0f);
        data[y * width * 4 + x * 4 + 1] = cast(ubyte)(normal.y * 255.0f);
        data[y * width * 4 + x * 4 + 2] = cast(ubyte)(normal.z * 255.0f);
        data[y * width * 4 + x * 4 + 3] = cast(ubyte)0;
      }
    }
    WriteDDS("normals.dds", width, height, DDSLoader.DXGI_FORMAT.R8G8B8A8_UNORM, (cast(void*)data.ptr)[0..(width * height * 4 * ubyte.sizeof)]);
    writefln("normals.dds written");
  }

  {
    auto data = NewArray!float(width * height * 4);
    scope(exit) Delete(data);
    for(uint y=0; y<height; y++)
    {
      for(uint x=0; x<width; x++)
      {
        data[y * width * 4 + x * 4] = pixels[y * width + x].directLight.x;
        data[y * width * 4 + x * 4 + 1] = pixels[y * width + x].directLight.y;
        data[y * width * 4 + x * 4 + 2] = pixels[y * width + x].directLight.z;
        data[y * width * 4 + x * 4 + 3] = 0.0f;
      }
    }
    WriteDDS("directLight.dds", width, height, DDSLoader.DXGI_FORMAT.R32G32B32A32_FLOAT, (cast(void*)data.ptr)[0..(width * height * 4 * float.sizeof)]);
    writefln("directLight.dds written");
  }
}

int main(string[] argv)
{
  thBase.asserthandler.Init();
  version(USE_SSE)
  {
    if(!core.cpuid.sse41)
    {
      writefln("Your processor does not support SSE 4.1 please use the non SSE version");
      return 1;
    }
  }

  thBase.asserthandler.Init();
  SDL.LoadDll("SDL.dll","libSDL-1.2.so.0");

  if(SDL.Init(SDL.INIT_VIDEO) < 0)
  {
    writefln("Initializing SDL failed");
    return 1;
  }
  scope(exit) SDL.Quit();

  SDL.Surface* screen = SDL.SetVideoMode(g_width, g_height, 32, SDL.HWSURFACE);

  if(screen is null)
  {
    writefln("creating screen surface failed");
    return 1;
  }

  auto timer = cast(shared(Timer))New!Timer();
  scope(exit) Delete(timer);

  auto loadSceneStart = Zeitpunkt(timer);
  loadScene();
  auto loadSceneEnd = Zeitpunkt(timer);
  writefln("loading the scene took %f s", (loadSceneEnd - loadSceneStart) / 1000.0f);
  allocThreadLocals();

  //allocate one element more for sse tone mapper
  Pixel[] pixels = NewArray!Pixel(g_width * g_height + 1)[0..$-1];
  scope(exit) Delete(pixels);
  SDL.Event event;

  /*
  // draw wireframe
  vec2 screenSize = vec2(cast(float)g_width, cast(float)g_height);
  foreach(edge; g_scene.textureEdges.objects)
  {
    drawLine(screen, edge.v[0] * screenSize, edge.v[1] * screenSize);
  }
  SDL.Flip(screen); */


  //rasterTriangles(39, 40, pixels);
  //rasterTriangles(2, 4, pixels);
  uint start = 0;
  uint triangleStep = 1000;
  while(true)
  {
    uint end = start + triangleStep;
    if(end > g_scene.triangles.length)
      end = g_scene.triangles.length;
    rasterTriangles(start, end, pixels);
    drawScreen(screen, pixels);
    start += triangleStep;
    if(g_scene.triangles.length == end)
      break;

    while(SDL.PollEvent(&event)) 
    {      
    }
  }

  auto rasterEnd = Zeitpunkt(timer);
  writefln("rastering triangles took %f milliseconds", (rasterEnd - loadSceneEnd) / 1000.0f);

  /*while(true)
  {
    while(SDL.PollEvent(&event)) 
    {      
    }
  }*/

  

  int h = 0;


  Random gen;

  writefln("computing sampling patterns...");
  precomputeSamples(gen);
  auto endComputeSamples = Zeitpunkt(timer);
  writefln("Computing sampling patterns took %f s", (endComputeSamples - rasterEnd) / 1000.0f);

  uint step = 64;
  uint steps = cast(uint)(pixels.length / step);

  PerPixelTask[] edgeTasks = NewArray!PerPixelTask(steps);
  auto edgeTaskIdentifier = TaskIdentifier.Create!"GeometryExtrapolation"();
  for(uint i=0; i < steps; i++)
  {
    auto startIndex = i * step;
    edgeTasks[i] = New!PerPixelTask(edgeTaskIdentifier, &extrapolateGeometry, startIndex, pixels[startIndex..startIndex+step]);
  }
  scope(exit)
  {
    foreach(task; edgeTasks)
      Delete(task);
    Delete(edgeTasks);
  }

  PerPixelTask[] tasks = NewArray!PerPixelTask(steps);
  auto taskIdentifier = TaskIdentifier.Create!"TakeSamples"();
  for(uint i=0; i < steps; i++)
  {
    auto startIndex = i * step;
    tasks[i] = New!PerPixelTask(taskIdentifier, &takeSamples, startIndex, pixels[startIndex..startIndex+step]);
  }
  scope(exit)
  {
    foreach(task; tasks)
      Delete(task);
    Delete(tasks);
  }

  PerPixelTask[] directLightTasks = NewArray!PerPixelTask(steps * 8);
  auto directLightTaskIdentifier = TaskIdentifier.Create!"DirectLight"();
  for(uint i=0; i < steps * 8; i++)
  {
    auto startIndex = i * step / 8;
    directLightTasks[i] = New!PerPixelTask(directLightTaskIdentifier, &computeDirectLight, startIndex, pixels[startIndex..startIndex+step]);
  }
  scope(exit)
  {
    foreach(task; directLightTasks)
      Delete(task);
    Delete(directLightTasks);
  }

  SmartPtr!(Worker)[] workers;

  auto threadsPerCPU = core.cpuid.threadsPerCPU;
  if(g_numThreads > threadsPerCPU)
  {
    g_numThreads = threadsPerCPU;
  }

  if(g_numThreads > 1)
  {
    workers = NewArray!(SmartPtr!Worker)(g_numThreads-1);
    for(uint i=0; i<g_numThreads-1; i++)
    {
      workers[i] = New!Worker(i+1);
      workers[i].start();
    }
  }
  scope(exit)
  {
    if(g_numThreads > 1)
      Delete(workers);
  }

  /*for(size_t i=0; i < g_height; i++)
  {
    takeSamples(pixels[g_width * i..g_width * (i+1)], gen);
    drawScreen(screen, pixels);
  }*/

  bool run = true;

  static if(extrapolateGeometryEnabled)
  {
    writefln("extrapolating geometry...");
    foreach(task; edgeTasks)
    {
      spawn(task);
    }

    while(!edgeTaskIdentifier.allFinished && run)
    //while(true)
    {
      g_localTaskQueue.executeOneTask();
      drawScreen(screen, pixels);

      while(SDL.PollEvent(&event)) 
      {      
        switch (event.type) 
        {
          case SDL.QUIT:
            run = false;
            break;
            /*case SDL.KEYDOWN:
            run = false;
            break;*/
          default:
        }
      }
    }
    if(!run)
    {
      g_run = false;

      foreach(worker; workers)
      {
        worker.join(false);
      }

      return 1;
    }
  }
  auto endGeometryExtrapolation = Zeitpunkt(timer);
  static if(extrapolateGeometryEnabled)
    writefln("extrapolating geometry took %f s", (endGeometryExtrapolation - endComputeSamples) / 1000.0f);

  pixels[g_width-1].rastered = false;
  pixels[g_width-1].position = vec3(-float.max);

  // compute direct light
  static if(computeDirectLightEnabled)
  {
    writefln("computing direct light..");
    foreach(task; directLightTasks)
    {
      spawn(task);
    }

    while(!directLightTaskIdentifier.allFinished && run)
      //while(true)
    {
      g_localTaskQueue.executeOneTask();
      drawScreen(screen, pixels);

      while(SDL.PollEvent(&event)) 
      {      
        switch (event.type) 
        {
          case SDL.QUIT:
            run = false;
            break;
            /*case SDL.KEYDOWN:
            run = false;
            break;*/
          default:
        }
      }
    }
    if(!run)
    {
      g_run = false;

      foreach(worker; workers)
      {
        worker.join(false);
      }

      return 1;
    }
  }
  auto endDirectLight = Zeitpunkt(timer);
  static if(computeDirectLightEnabled)
    writefln("computing direct light took %f s", (endDirectLight - endGeometryExtrapolation) / 1000.0f);

  // take samples
  writefln("computing sample locations...");
  foreach(task; tasks)
  {
    spawn(task);
  }

  while(!taskIdentifier.allFinished && run)
  {
    g_localTaskQueue.executeOneTask();
    drawScreen(screen, pixels);

    while(SDL.PollEvent(&event)) 
    {      
      switch (event.type) 
      {
        case SDL.QUIT:
          run = false;
          break;
          /*case SDL.KEYDOWN:
          run = false;
          break;*/
        default:
      }
    }
  }
  if(!run)
  {
    g_run = false;

    foreach(worker; workers)
    {
      worker.join(false);
    }

    return 1;
  }

  auto endTakeSamples = Zeitpunkt(timer);
  writefln("Computing samples and ambient took %f s", (endTakeSamples - endDirectLight) / 1000.0f);

  writeDDSFiles(g_width, g_height, pixels);
  auto endWriteData = Zeitpunkt(timer);
  writefln("writing data took %f s", (endWriteData - endTakeSamples) / 1000.0f);
  writefln("total time taken %f s", (endWriteData - loadSceneStart) / 1000.0f);

  uint totalSamples = 0;
  uint wastedSamples = 0;
  uint backfaceSamples = 0;
  foreach(ref data; g_perWorkerData)
  {
    totalSamples += data.totalSamples;
    wastedSamples += data.wastedSamples;
    backfaceSamples += data.backfaceSamples;
  }
  writefln("%.2f%% of all samples wasted", cast(float)wastedSamples / cast(float)totalSamples * 100.0f);
  writefln("%.2f%% of all samples hit a backface", cast(float)backfaceSamples / cast(float)totalSamples * 100.0f);

  /*uint progress = 0;
  uint step = g_width * 4;
  uint steps = cast(uint)(pixels.length / step);
  PerPixelTask[] tasks = NewArray!PerPixelTask(steps);
  auto taskIdentifier = TaskIdentifier.Create!"PerPixelTask"();
  for(uint i=0; i < steps; i++)
  {
    auto startIndex = i * step;
    tasks[i] = New!PerPixelTask(taskIdentifier, startIndex, pixels[startIndex..startIndex+step]);
  }
  scope(exit)
  {
    foreach(task; tasks)
      Delete(task);
    Delete(tasks);
  }
  SmartPtr!(Worker)[] workers;

  auto threadsPerCPU = core.cpuid.threadsPerCPU;
  if(g_numThreads > threadsPerCPU)
  {
    g_numThreads = threadsPerCPU;
  }

  if(g_numThreads > 1)
  {
    workers = NewArray!(SmartPtr!Worker)(g_numThreads-1);
    for(uint i=0; i<g_numThreads-1; i++)
    {
      workers[i] = New!Worker();
      workers[i].start();
    }
  }
  scope(exit)
  {
    if(g_numThreads > 1)
      Delete(workers);
  }*/

  float totalTime = 0.0f;

  auto startRendering = Zeitpunkt(timer);
  auto startPass = startRendering;
  while(run)
  {
    while(SDL.PollEvent(&event)) 
    {      
      switch (event.type) 
      {
        case SDL.QUIT:
          run = false;
          break;
        default:
      }
    }
  }

  /*while(!taskIdentifier.allFinished)
    g_localTaskQueue.executeOneTask();*/

  g_run = false;

  foreach(worker; workers)
  {
    worker.join(false);
  }

  auto endRendering = Zeitpunkt(timer);
  SDL.Quit();

  core.stdc.stdlib.system("pause");

  return 0;
}
