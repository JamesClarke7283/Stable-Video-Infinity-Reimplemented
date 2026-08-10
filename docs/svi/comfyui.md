### [10 Dec 2025] Important Notes on Using LightX2V with SVI 2.0

**New Known Issue.** We just found that the LightX2V LoRA strength has a significant impact to SVI. With the original value of 1.0, we see clearly worse dynamics and text-following compared to runs with a reduced strength in specific samples. However, lowering the LightX2V LoRA weight will also affect the generation quality when using 4/8-step sampling. So this may be a trade-off: **it’s better to try reducing this value first if seeing poor text-following, weak dynamics, or the reference frame reappearing.** More comparisons can be found below.

PS: The reason we use this sample for testing is that it was originally a failure case reported by a user. :)

https://github.com/user-attachments/assets/bbf9be6c-b357-474f-acd5-8e617d18c8e8

**Current Solution.** The current optimal setup we’ve found so far is to reduce the LightX2V LoRA strength from 1.0 to 0.5 for the high-noise stages (which mainly control large motions), while keeping it at 1.0 for the low-noise stages. Please check out our updated comfyui workflow. If anyone has found a better solution, please let us know! Thank you so much!

In the example below, for the high-noise model, the left sample uses a LightX2V scale of 0.5 and the right uses 0.6. We can see that the motion increases as the scale decreases, which is consistent with the ablation results above. 

<table>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/b2159a91-8b90-444c-aabc-1cf2717bbd96"
             controls
             muted
             width="100%">
      </video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/adb92690-c024-47e2-aa62-5e9b5e718389"
             controls
             muted
             width="100%">
      </video>
    </td>
  </tr>
</table>

### [26 Dec 2025] Important Notes on Using LightX2V with SVI 2.0 Pro

SVI 2.0 Pro does not support previous workflows. 