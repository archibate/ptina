# Tina

A path tracer based on [Taichi](https://github.com/taichi-dev/taichi)

## How to run

For Linux & Mac OS X:
```py
python3 -m pip -U -r requirements.txt
export PYTHONPATH=`pwd`
python3 exams/interactive.py
```

For Windows 10:
```py
python -m pip -U -r requirements.txt
python -c "import os, sys, runpy; sys.path.append(os.getcwd()); runpy.run_path('exams/interactive.py')"
```

## Features

- Disney BSDF (with transmission)
- interactive camera control
- linear BVH for acceleration
- point, area, and environment lights
- textures, and a memory allocator for it
- multiple importance sampling (light & BSDF)
- Sobol quasi-random sequence generator
- metropilis light transport path tracing
- albedo & normal rendering as AOV
- Blender intergration as addon

## Performance

- To be tested

## References

https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf
http://www.pbr-book.org/3ed-2018/Reflection_Models/Microfacet_Models.html#eq:beckmann-lambda
https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/
https://web.maths.unsw.edu.au/~fkuo/sobol/
