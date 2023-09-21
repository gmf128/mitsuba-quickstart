import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt

mi.set_variant("cuda_ad_rgb")
scene = mi.load_file("../scenes/shader_ball/scene.xml", spp=512)
params = mi.traverse(scene)

img = mi.render(scene)
img = mi.util.convert_to_bitmap(img)

plt.axis("off")
plt.imshow(img)
plt.savefig("rough_dielectric_2.png")