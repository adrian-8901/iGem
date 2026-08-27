import numpy as np
import trimesh

# 1. Sphere (Minimal surface-area-to-volume ratio / baseline control)
sphere = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
sphere.export("sphere.stl")

# 2. Capsule / Pill (Standard oral dosage geometry)
capsule = trimesh.creation.capsule(height=10.0, radius=3.0)
capsule.export("capsule.stl")

# 3. Torus / Ring (High surface-area-to-volume ratio for rapid core exposure)
torus = trimesh.creation.torus(major_radius=7.0, minor_radius=2.0)
torus.export("torus.stl")

print("Generated: sphere.stl, capsule.stl, torus.stl")
