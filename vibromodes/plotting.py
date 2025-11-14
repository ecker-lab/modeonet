import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import torch
from vibromodes.kirchhoff import tr_velocity_field_to_frequency_response

def plot_grad_flow(plot_name,named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if(p.grad is None):
                print(n,p.shape)
            ave_grads.append(p.grad.abs().mean().cpu())
    #print(ave_grads)
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.yscale("log")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    #plt.grid(True)
   

@torch.no_grad()
def compare_fields_side_by_side(pred_field, tgt_field, interval=50, save_path="comparison.mp4"):
    """
    Compare two videos side by side and save the result as an MP4 file.

    Parameters:
    - video1, video2: torch.tensor of shape (300, height, width)
    - interval: delay between frames in milliseconds
    - save_path: filename for the output video
    """
    print("test")
    assert pred_field.shape[0] == tgt_field.shape[0], "Videos must have the same number of frames"


    tgt_frf = tr_velocity_field_to_frequency_response(tgt_field)
    pred_frf = tr_velocity_field_to_frequency_response(pred_field)
    
    num_frames = pred_field.shape[0]
    
    fig, axs = plt.subplots(3,1)

    axs[0].plot(tgt_frf.detach().cpu().numpy(),label="tgt")
    axs[0].plot(pred_frf.detach().cpu().numpy(),linestyle="--",label="pred")
    axs[0].legend()

    (line,) = axs[0].plot([0,0],[0,60],color="grey")


    axs[1].axis("off")

    pred_field = pred_field.detach().cpu().numpy()
    tgt_field = tgt_field.detach().cpu().numpy()

    combined = np.concatenate((np.absolute(pred_field[0]), np.absolute(tgt_field[0])), axis=1)
    combined = (combined-combined.min())/(combined.max()-combined.min())

    im = axs[1].imshow(combined)

    axs[1].set_title("pred \t tgt")

    combined = np.concatenate((np.angle(pred_field[0]), np.angle(tgt_field[0])), axis=1)
    combined = (combined-combined.min())/(combined.max()-combined.min())

    im_angle = axs[2].imshow(combined)

    axs[2].axis("off")
    plt.tight_layout()


    def update(frame):
        combined = np.concatenate((np.absolute(pred_field[frame]), 
                                   np.absolute(tgt_field[frame])), axis=1)
        combined = (combined-combined.min())/(combined.max()-combined.min())
        im.set_array(combined)


        combined = np.concatenate((np.angle(pred_field[frame]), 
                                   np.angle(tgt_field[frame])), axis=1)
        combined = (combined-combined.min())/(combined.max()-combined.min())
        im_angle.set_array(combined)

        line.set_xdata([frame,frame])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True)

    # Save the animation as an MP4
    ani.save(save_path, writer='ffmpeg', fps=1000//interval)

    plt.show()

