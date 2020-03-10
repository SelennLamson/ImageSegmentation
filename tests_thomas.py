import image_segmentation as seg


def seg_func_dummy(img, scrib):
	return scrib


gui = seg.Gui(segmentation_function=seg_func_dummy)
gui.start()

