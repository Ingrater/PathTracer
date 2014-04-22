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
import core.thread;
static import core.cpuid;

import std.math;
import core.stdc.math : cosf, sinf;
import core.stdc.stdlib;

import sdl;
import rendering;
import scene;

//version = PerformanceTest;

void setPixel(SDL.Surface *screen, int x, int y, ubyte r, ubyte g, ubyte b)
{
  uint *pixmem32;
  uint colour;  

  colour = SDL.MapRGB( screen.format, r, g, b );

  pixmem32 = cast(uint*)(screen.pixels + (y * screen.pitch + x * 4));
  *pixmem32 = colour;
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
class ComputeOutputTask : Task
{
  private:
    uint m_pixelOffset;
    Pixel[] m_pixels;
    Random m_gen;

  public:
    this(TaskIdentifier identifier, uint pixelOffset, Pixel[] pixels)
    {
      super(identifier);
      m_pixelOffset = pixelOffset;
      m_pixels = pixels;
      m_gen.seed(m_pixelOffset + cast(uint)pixels[0].n);
    }

    override void Execute()
    {
      //computeOutputColor(m_pixelOffset, m_pixels, m_gen);
      takeSamples(m_pixels, m_gen);
    }

    override void OnTaskFinished() {}
}

// Worker thread
class Worker : Thread
{
  this()
  {
    super(&run);
  }

  void run()
  {
    g_localTaskQueue.executeTasksUntil( (){ return !g_run; } );
  }
}

auto interpolate(T)(float u, float v, T val0, T val1, T val2)
{
  float x = 1.0f / (u + v);
  float u1 = x * u;
  float v1 = x * v;
  immutable float sqrt2 = 1.414213562f;
  float d1 = fastsqrt((1.0f-u1)*(1.0f-u1) + v1*v1) / sqrt2;
  float d2 = fastsqrt(u1*u1 + (1.0f-v1)*(1.0f-v1)) / sqrt2;
  auto interpolated1 = val1 * d1 + val2 * d2;

  float len = fastsqrt(u1*u1 + v1*v1);
  float i1 = fastsqrt(u*u+v*v) / len;
  float i2 = 1.0f - i1;

  return val0 * i2 + interpolated1 * i1;
}

float g_cylinderRadius = 0.50f;

vec3 mapToCylinder(vec2 p)
{
  return vec3(cosf(p.x * PI * 2.0) * g_cylinderRadius, sinf(p.x * PI * 2.0) * g_cylinderRadius, p.y);
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
    assert(!epsilonCompare(y1, y2));

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

    vec2 vU = verts[1] - verts[0];
    vec2 vV = verts[2] - verts[0];

    vec2[3] uvs;
    uvs[0] = vec2(0.0f, 0.0f);
    uvs[1] = vec2(0.0f, 1.0f);
    uvs[2] = vec2(1.0f, 0.0f);

    uint[3] swizzle = [0, 1, 2];
    vec3[3] wsPos = [ triangle.v0, triangle.v1, triangle.v2 ];

    if(verts[0].y > verts[1].y)
    {
      swap(verts[0], verts[1]);
      swap(uvs[0], uvs[1]);
      swap(swizzle[0], swizzle[1]);
    }
    if(verts[0].y > verts[2].y)
    {
      swap(verts[1], verts[2]);
      swap(verts[0], verts[1]);
      swap(uvs[1], uvs[2]);
      swap(uvs[0], uvs[1]);
      swap(swizzle[1], swizzle[2]);
      swap(swizzle[0], swizzle[1]);
    }
    else if(verts[1].y > verts[2].y)
    {
      swap(verts[1], verts[2]);
      swap(uvs[1], uvs[2]);
      swap(swizzle[1], swizzle[2]);
    }

    Edge[3] edges;
    uint numEdges = 0;

    /*vU = vec2(vU.x * fWidth, vU.y * fHeight);
    vV = vec2(vV.x * fWidth, vV.y * fHeight);
    vec2 uvDelta = vec2(vU.x / vU.dot(vU), vV.x / vV.dot(vV));*/

    if(cast(uint)(verts[0].y * fHeight) != cast(uint)(verts[1].y * fHeight))
    {
      edges[numEdges++] = Edge(verts[0].x * fWidth, verts[0].y * fHeight, uvs[0], verts[1].x * fWidth, verts[1].y * fHeight, uvs[1]);
      edges[numEdges++] = Edge(verts[0].x * fWidth, verts[0].y * fHeight, uvs[0], verts[2].x * fWidth, verts[2].y * fHeight, uvs[2]);
      if(cast(uint)(verts[1].y * fHeight) != cast(uint)(verts[2].y * fHeight))
      {
        edges[numEdges++] = Edge(verts[1].x * fWidth, verts[1].y * fHeight, uvs[1], verts[2].x * fWidth, verts[2].y * fHeight, uvs[2]);
      }
    }
    else
    {
      edges[numEdges++] = Edge(verts[0].x * fWidth, verts[0].y * fHeight, uvs[0], verts[2].x * fWidth, verts[2].y * fHeight, uvs[2]);
      edges[numEdges++] = Edge(verts[1].x * fWidth, verts[1].y * fHeight, uvs[1], verts[2].x * fWidth, verts[2].y * fHeight, uvs[2]);
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
      for(uint x = start; x < end; x++)
      {
        uv += uvDelta;
        auto curPixel = &pixels[y * g_width + x];
        curPixel.rastered = true;
        curPixel.position = interpolate(uv.x, uv.y, wsPos[0], wsPos[1], wsPos[2]);
        curPixel.normal = interpolate(uv.x, uv.y, t.n[0], t.n[1], t.n[2]);
        vec2 coords = interpolate(uv.x, uv.y, t.tex[0], t.tex[1], t.tex[2]);
        //curPixel.color = curPixel.position * (1.0f / 20.0f) + vec3(0.5f);
        //curPixel.color = vec3(coords.x, coords.y, 0.0f);
        //curPixel.color = vec3(uv.x, uv.y, 0.0f);
        curPixel.color = curPixel.normal * 0.5f + vec3(0.5f);
        
      }
      edges[0].xs += edges[0].invM;
      edges[0].uv += edges[0].uvDelta;
      edges[1].xs += edges[1].invM;
      edges[1].uv += edges[1].uvDelta;
    }
  }
}

void takeSamples(Pixel[] pixels, ref Random gen)
{
  vec2[] pattern = (cast(vec2*)alloca(vec2.sizeof * Pixel.samples.length))[0..Pixel.samples.length];
  foreach(ref pixel; pixels)
  {
    if(!pixel.rastered)
      continue;
    bestCanidatePattern!(minDistCylinder)(pattern, gen);
    foreach(size_t i, ref sample; pixel.samples)
    {
	    float psi = pattern[i].x  * 2.0f * PI; //uniform(0, 2 * PI, gen);
	    float phi = (pattern[i].y * 0.9f + 0.1f) * PI_2; //uniform(0, PI_2, gen);
	    vec3 sampleDir = angleToDirection(phi, psi, pixel.normal);
	    Ray sampleRay = Ray(pixel.position + pixel.normal * 0.1f, sampleDir);
	    float hitDistance = 0.0f;
	    vec2 hitTexcoords;
	    const(Scene.TriangleData)* hitData;
	    if( g_scene.trace(sampleRay, hitDistance, hitTexcoords, hitData))
      {
		    sample = hitTexcoords;
	    }
      else
      {
        sample = vec2(0.0f, 0.0f); // nothing hit
      }
      if(i==1)
      {
        pixel.color = vec3(hitTexcoords.x, hitTexcoords.y, 0.0f);
      }
    }
  }
}

void writeDDSFiles(uint width, uint height, Pixel[] pixels)
{
  auto data = NewArray!ushort(width * height * 4);
  scope(exit) Delete(data);
  uint numFiles = (Pixel.samples.length + 1) / 2;
  for(uint i=0; i<numFiles; i++)
  {
    for(uint y=0; y<width; y++)
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

int main(string[] argv)
{
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

  loadScene();
  allocThreadLocals();

  //allocate one element more for sse tone mapper
  Pixel[] pixels = NewArray!Pixel(g_width * g_height + 1)[0..$-1];
  scope(exit) Delete(pixels);

  //rasterTriangles(39, 40, pixels);
  //rasterTriangles(2, 4, pixels);
  rasterTriangles(0, g_scene.triangles.length, pixels);
  drawScreen(screen, pixels);

  

  int h = 0;
  SDL.Event event;

  Random gen;

  uint step = g_width;
  uint steps = cast(uint)(pixels.length / step);
  ComputeOutputTask[] tasks = NewArray!ComputeOutputTask(steps);
  auto taskIdentifier = TaskIdentifier.Create!"ComputeOutputTask"();
  for(uint i=0; i < steps; i++)
  {
    auto startIndex = i * step;
    tasks[i] = New!ComputeOutputTask(taskIdentifier, startIndex, pixels[startIndex..startIndex+step]);
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
  }

  foreach(task; tasks)
  {
    spawn(task);
  }

  /*for(size_t i=0; i < g_height; i++)
  {
    takeSamples(pixels[g_width * i..g_width * (i+1)], gen);
    drawScreen(screen, pixels);
  }*/

  bool run = true;
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

  writeDDSFiles(g_width, g_height, pixels);

  /*uint progress = 0;
  uint step = g_width * 4;
  uint steps = cast(uint)(pixels.length / step);
  ComputeOutputTask[] tasks = NewArray!ComputeOutputTask(steps);
  auto taskIdentifier = TaskIdentifier.Create!"ComputeOutputTask"();
  for(uint i=0; i < steps; i++)
  {
    auto startIndex = i * step;
    tasks[i] = New!ComputeOutputTask(taskIdentifier, startIndex, pixels[startIndex..startIndex+step]);
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

  auto timer = cast(shared(Timer))New!Timer();
  scope(exit) Delete(timer);

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
  writefln("Rendering took %f seconds", (endRendering - startRendering) / 1000.0f);

  core.stdc.stdlib.system("pause");

  return 0;
}
