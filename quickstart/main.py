import mitsuba as mi
import drjit as jit
import time
"""
tutorial 1: quick start: how to load a scene and render
"""
# set variant
mi.set_variant('cuda_ad_rgb')

# load scene
start = time.time()
scene = mi.load_file("../scenes/cbox.xml")

# render
img = mi.render(scene, spp=256)

# output the rendered image
mi.util.write_bitmap("results/cbox.png", img)
end = time.time()
print(end - start)
"""
tutorial 2: how to edit a scene using api
"""
params = mi.traverse(scene)
##print(params)
# double the emitter`s radiance
params['light.emitter.radiance.value'] *= 2
params.update()

img2 = mi.render(scene, spp=256)
mi.util.write_bitmap("results/cbox_2.png", img2)

