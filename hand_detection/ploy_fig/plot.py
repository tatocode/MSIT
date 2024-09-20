import os
import matplotlib.pyplot as plt
import cv2

if __name__ == "__main__":
  true_file = "true"
  false_file = "false"

  true_img_paths = [os.path.join(true_file, name) for name in os.listdir(true_file)]
  false_img_paths = [os.path.join(false_file, name) for name in os.listdir(false_file)]

  # plt.figure(figsize=(24, 7))
  # for ix in range(len(true_img_paths)):
  #   plt.subplot(2, 4, ix+1)
  #   plt.axis("off")
  #   plt.subplots_adjust(wspace=0, hspace =0)
  #   plt.imshow(cv2.imread(true_img_paths[ix])[:, :, ::-1])
  # plt.tight_layout()
  # plt.savefig("true.png")

  plt.figure(figsize=(15, 9))
  for ix in range(len(false_img_paths)):
    plt.subplot(2, 2, ix+1)
    plt.axis("off")
    plt.subplots_adjust(wspace=0, hspace =0)
    plt.imshow(cv2.imread(false_img_paths[ix])[:, :, ::-1])
  plt.tight_layout()
  plt.savefig("false.png")