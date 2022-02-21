import json
import sys
import numpy
import torch
import torchvision
import torchvision.transforms as tvtf
from PIL import Image
from tqdm import tqdm

# Rather than get really fancy and define a Dataset, let's just use the built-in torchvision.datasets.DatasetFolder.
# DatasetFolder takes a top-level dataset root/(class_a | class_b | ...)/*.jpg and returns a list of (img, class).
# We only care if classes are or are not the same at train time.

LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_WORKERS = 2
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(width:int, height:int, embedding_size:int):
	return torch.nn.Sequential(
		torch.nn.Flatten(),
		torch.nn.Linear(in_features=width*height, out_features=256),
		torch.nn.LeakyReLU(inplace=True),
		torch.nn.Linear(256, 128),
		torch.nn.LeakyReLU(inplace=True),
		torch.nn.Linear(128, 32),
		torch.nn.LeakyReLU(inplace=True),
		torch.nn.Linear(32, embedding_size),
		# No activation on last layer?
	)


def main(training_data_directory, saved_model_directory):
	model = build_model(32, 32, 10).to(DEVICE)

	# Set up some rescaling and random flips to give us data augmentation, but don't add noise.
	transforms = tvtf.Compose([
		tvtf.Grayscale(),
		tvtf.RandomHorizontalFlip(),
		tvtf.RandomVerticalFlip(),
		tvtf.RandomRotation(20),
		# We don't random-resize or random-crop because all of our samples are bound and based on our input size.
		#tvtf.Resize((40, 40)),
		#tvtf.RandomCrop((32, 32)),
		tvtf.ToTensor(),  # Converts 0,255 PIL -> 0.0,1.0 Tensor.
	])

	# Brace for run...
	loss_fn = torch.nn.CosineEmbeddingLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
	dataset = torchvision.datasets.ImageFolder(training_data_directory, transform=transforms)
	dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

	# Training loop:
	for epoch_idx in range(EPOCHS):
		dataloop = tqdm(dataset_loader)
		total_epoch_loss = 0.0
		for batch_idx, (data, targets) in enumerate(dataloop):
			step = (epoch_idx * len(dataloop)) + batch_idx
			data = data.to(device=DEVICE)
			optimizer.zero_grad()

			# Forward
			embeddings = model(data)

			# One embedding gives us n*(n-1) pairs of datapoints.
			# We rely on the batch being shuffled and having some of each class, but if the entire batch is unlucky
			# and we have all one class, it will be okay.
			# left takes [1, 2, 3, 4] and goes to [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
			# right takes [1, 2, 3, 4] and goes to [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
			left = torch.repeat_interleave(embeddings, embeddings.shape[0], axis=0)
			right = embeddings.repeat(embeddings.shape[0], 1)
			truth = list()
			for label_left in targets:
				for label_right in targets:
					truth.append(1.0 if label_left == label_right else -1.0)
			truth = torch.tensor(truth).to(DEVICE)

			# Embedding pairs are 1 if they're the same and -1 if they're not.
			# We match up embeddings based on their classes.
			loss = loss_fn(left, right, truth)

			# Backward
			loss.backward()
			optimizer.step()

			# Log status.
			total_epoch_loss += loss.item()

		print(f"Total epoch loss: {total_epoch_loss}")
		torch.save(model.state_dict(), f"checkpoints/checkpoint_{epoch_idx}")
	save_model_to_json(model)
	torch.save(model, "result_model.pt")


def save_model_to_json(model):
	result_model = dict()
	result_model['description'] = model.__str__()
	result_model['weights'] = list()
	result_model['biases'] = list()
	result_model['shapes'] = list()
	for layer_idx in range(len(model)):
		for param_idx, param in enumerate(model[layer_idx].parameters()):
			weight_or_bias = param.to('cpu').detach().numpy().T
			if len(weight_or_bias.shape) == 1:
				result_model['biases'].append([float(x) for x in weight_or_bias.flatten()])
			else:
				result_model['weights'].append([float(x) for x in weight_or_bias.flatten()])
				result_model['shapes'].append(weight_or_bias.shape)

	with open("result_model.json", 'wt') as fout:
		json.dump(result_model, fout)

# Utility method for getting embeddings.
def embed(filename, model):
	img = Image.open(filename)
	tnsor = torch.Tensor(numpy.asarray(img)[:,:,0]/255.0).unsqueeze(0)
	return model(tnsor)[0]


if __name__=="__main__":
	main(sys.argv[1], sys.argv[2])