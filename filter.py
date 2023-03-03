
# TODO
# Nanobio zoliból kivágva, még keretbe kell önteni!


# frame = well[-1, :, :].copy()
# med = np.median(frame)
# sliders = np.array([med, 0.5])
# frame[np.logical_and(frame < sliders[0] + sliders[1] / 2, frame > sliders[0] - sliders[1] / 2)] = 1000
# fig, ax = plt.subplots(figsize=(20, 10))
# elm = ax.imshow(frame, vmin=med - 1, vmax= med + 1)

# def change_frame():
# #     del frame
#     frame = well[-1, :, :].copy()
#     frame[np.logical_and(frame < sliders[0] + sliders[1] / 2, frame > sliders[0] - sliders[1] / 2)] = 1000
#     elm.set_data(frame)
    
# def slider_mean_change(s):
#     sliders[0] = float(s['new'])
#     change_frame()

# def slider_width_change(s):
#     sliders[1] = float(s['new'])
#     change_frame()

# slider_mean = widgets.FloatSlider(description="Mean Slider",
#                                value = med, min = med-2, max= med + 2, 
#                                 step = 0.1)
# slider_width = widgets.FloatSlider(description="Width Slider",
#                                   value = 0.5, min = 0, max= 2, 
#                                 step = 0.02)
# box = widgets.VBox([slider_mean, slider_width])
# slider_mean.observe(slider_mean_change, names='value')
# slider_width.observe(slider_width_change, names='value')
# display(box)