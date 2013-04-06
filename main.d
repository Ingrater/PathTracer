module main;

import std.random;
import thBase.io;
import thBase.math;
import thBase.asserthandler;
import thBase.timer;
import thBase.task;
import core.thread;

import sdl;
import rendering;

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
  version(USE_SSE)
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
        asm {
          mov EAX, p;
          movups XMM3, [EAX];
          minps XMM3, XMM0; //min(x, 1)
          maxps XMM3, XMM1; //max(x, 0)
          mulps XMM3, XMM2; //x *= 255.0f
          cvtps2dq XMM3, XMM3; //convert to int
          lea EAX, colors;
          movups [EAX], XMM3;
        }
        setPixel(screen, x, y, cast(ubyte)colors[0], cast(ubyte)colors[1], cast(ubyte)colors[2]);
      }
    }
  }
  else
  {
    for(int y = 0; y < screen.height; y++ ) 
    {
      for(int x = 0; x < screen.width; x++ ) 
      {
        Pixel* p = &pixels[g_width * y + x];
        ubyte r = cast(ubyte)(saturate(p.color.r) * 255.0f);
        ubyte g = cast(ubyte)(saturate(p.color.g) * 255.0f);
        ubyte b = cast(ubyte)(saturate(p.color.b) * 255.0f);
        setPixel(screen, x, y, r, g, b);
      }
    }
  }

  SDL.Flip(screen); 
}

__gshared bool g_run = true;

class ComputeOutputTask : Task
{
  private:
    uint m_pixelOffset;
    Pixel[] m_pixels;
    Random m_gen;
    uint m_run;

  public:
    this(TaskIdentifier identifier, uint pixelOffset, Pixel[] pixels)
    {
      super(identifier);
      m_pixelOffset = pixelOffset;
      m_pixels = pixels;
      m_run = 0;
    }

    override void Execute()
    {
      m_gen.seed(m_pixelOffset + m_run);
      m_run += 1337;
      computeOutputColor(m_pixelOffset, m_pixels, m_gen);
    }

    override void OnTaskFinished() {}
}

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

int main(string[] argv)
{
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

  //allocate one element more for sse tone mapper
  Pixel[] pixels = NewArray!Pixel(g_width * g_height + 1)[0..$-1];
  scope(exit) Delete(pixels);

  int h = 0;
  SDL.Event event;

  Random gen;

  uint progress = 0;
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

  auto timer = cast(shared(Timer))New!Timer();
  scope(exit) Delete(timer);

  float totalTime = 0.0f;

  while(g_run)
  {
    // one time rendering
    /*if(progress < steps)
    {
      auto start = Zeitpunkt(timer);
      auto startIndex = progress * step;
      computeOutputColor(startIndex, pixels[startIndex..startIndex+step], gen);
      drawScreen(screen, pixels);
      progress++;
      auto end = Zeitpunkt(timer);
      totalTime += (end - start) / 1000.0f;
      writefln("progress %d", progress);
      if(progress == steps)
        writefln("timeTaken %f", totalTime);
    }*/

    // scanline rendering
    /*{
      auto startIndex = progress * step;
      computeOutputColor(startIndex, pixels[startIndex..startIndex+step], gen);
      drawScreen(screen, pixels);
      progress++;
      if(progress >= steps)
        progress = 0;
    }*/

    // task based rendering
    {
      foreach(task; tasks)
      {
        spawn(task);
      }
      g_localTaskQueue.executeTasks();
      while(!taskIdentifier.allFinished) { } //spin lock
      drawScreen(screen, pixels);
    }

    while(SDL.PollEvent(&event)) 
    {      
      switch (event.type) 
      {
        case SDL.QUIT:
          g_run = false;
          break;
        /*case SDL.KEYDOWN:
          run = false;
          break;*/
        default:
      }
    }
  }

  foreach(worker; workers)
  {
    worker.join(false);
  }

  return 0;
}
