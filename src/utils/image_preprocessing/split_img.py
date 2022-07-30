import numpy as np
import os
from PIL import Image

def split_img(img_filename, save=False, patch_split=4, path='./'):
	"""
	img = Image.open(img_filename).convert('RGB')
	W, H = img.size ## should be 256, 256
	nbr_img = 16 ## should be a perfect square
	patches = []
	## test : reconstructing the original img (comment if unnecessary)
	reconstructed_img = Image.new('RGB', (W, H), color = (0, 0, 0))
	for index in range(nbr_img):
		i, j = int(W*(index//np.sqrt(nbr_img))/np.sqrt(nbr_img)), int(H*(index%np.sqrt(nbr_img))/np.sqrt(nbr_img)) ## left upper corner coordinates
		new_i, new_j = int(i + W/np.sqrt(nbr_img)) , int(j + H/np.sqrt(nbr_img)) ## right bottom corner coordinates
		area = (i, j, new_i, new_j)
		cropped_img = img.crop(area) ## cropping
		patches.append(np.asarray(cropped_img))

		if save:
			cropped_img.save(new_dir + '/' + str(index) + '.jpeg') ## saving

		Image.Image.paste(reconstructed_img, cropped_img, (int(i), int(j)))

	## test : reconstructing the original img (comment if unnecessary)
	if save:
		new_dir = img_filename[:-5]
		os.mkdir(new_dir)
		img.save(new_dir + '/original.jpeg')
		reconstructed_img.save(new_dir + '/reconstructed.jpeg')
	return None
	"""
	
	im = np.asarray(Image.open(img_filename))
	M = N = im.shape[0] // patch_split

	# Source: https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python
	tiles = [im[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]
	if save:
		for i, tile in enumerate(tiles):
			image = Image.fromarray(tile)
			image.save(os.path.join(path, f'{str(i)}.png'))

	return tiles
	

if __name__ == '__main__':
	## test with squared source img
	img_filename = 'squared_test_img.jpeg'
	split_img(img_filename)

	## test with non-squared source img
	img_filename = 'non_squared_test_img.jpeg'
	split_img(img_filename)