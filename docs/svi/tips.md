
## ⚠️ Tips for Generating Better Long Videos

**For improved textual alignment, you may consider the following crtical aspects.**

1. **Slightly increase the CFG value**  
   Try slightly increasing the CFG value within the recommended range of **1–2** to strengthen how much the model follows the text. The cfg value in the original inference setting is typically 5. However, due to the acceleration components integrated within the ComfyUI workflow, the cfg value cannot be maintained consistently. Consequently, we have adjusted and fixed the cfg at 1.5. 

2. **Prompt enhancement**  
   There is an inherent trade-off between the control signal from the text and from the reference frame. If you don’t give the model **clear and strong enough prompts**, it will **by default** follow the pose/posture in the reference frame. To address this, you can refine and strengthen your text prompts (prompt enhancement). As shown in the demo3 video, *“Your dog can run anywhere,”* when the prompts are well designed, the model can produce relatively large motion/dynamics **without snapping back to the reference frame**. Many thanks to this [Issue](https://github.com/vita-epfl/Stable-Video-Infinity/issues/35#issuecomment-3632223811) for highlighting this!


3. **[12-09-2025] Check and match the aspect ratio**  
   Since our model is trained at **480×832 (horizontal)**, it does **not** perform very well in text-following when you use very different aspect ratios (e.g., vertical), especially when the frame is filled with a full-body person (though the stability is still good). This often leads to weaker motion/dynamics and a stronger tendency to “snap back” to the reference image. Therefore, it’s better to **outpaint the input image to 480×832** (or a very similar horizontal ratio). This adjustment can make a significant difference in both motion and text-following quality. In practice, as discussed in the [Issue](https://github.com/vita-epfl/Stable-Video-Infinity/issues/35#issuecomment-3633842068), outpainting the reference image to match the training aspect ratio greatly reduces reference-image reappearing and improves overall behavior.

4. **Use high-quality input image**
   It’s better to ensure the first frame is high quality, since it also serves as the anchor that guides all subsequent generations.

The following demo compares the results generated with cfg=1 and cfg=2 (one-shot generation without cherry-picking).

<table>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/198aeeeb-a719-49a0-8977-3f1b9f3a9e47"
             controls
             muted
             width="100%">
      </video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/16709d84-fd05-4c7f-a1e1-633a856e3dbb"
             controls
             muted
             width="100%">
      </video>
    </td>
  </tr>
</table>

The following demo compares the results generated with origin (worse text-following and reference-image returning) and new aspect ratios (one-shot generation without cherry-picking).

<table>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/d4df2ee8-ec64-4566-913c-1ea9d73209a6"
             controls
             muted
             width="100%">
      </video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/eb436248-2636-445d-851a-de2a66986f9c"
             controls
             muted
             width="100%">
      </video>
    </td>
  </tr>
</table>



It is necessary to install `flash_attn`; otherwise, severe artifacts and instability will appear. The official Diffsynth installation does not include this step, so we are sorry for missing this at first. See details below. Left: without `flash_attn` &nbsp;&nbsp;|&nbsp;&nbsp; Right: with `flash_attn`. Without `flash_attn`, noticeable artifacts appear around the mouth at **16 seconds**.

<table>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/636c24dd-2bc2-4427-a920-a6bac99c33d1"
             controls
             muted
             width="100%">
      </video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/c81dc495-55f6-422c-964e-aa7e98826460"
             controls
             muted
             width="100%">
      </video>
    </td>
  </tr>
</table>
