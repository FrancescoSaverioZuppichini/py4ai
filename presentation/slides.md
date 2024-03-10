---
theme: default
# background: https://source.unsplash.com/collection/94734566/1920x1080
class: text-center
highlighter: shiki
lineNumbers: true
info: |
  ## Slidev Starter Template
  Presentation slides for developers.

  Learn more at [Sli.dev](https://sli.dev)
drawings:
  persist: false
defaults:
  foo: true
transition: slide-left
title: How to Efficiently Send Images to GPU 📸 🚀
mdc: true
monaco: true
monacoTypesSource: local # or cdn or none
fonts:
  # Roboto for general text, offering a clean and modern sans-serif look
  sans: "Roboto"
  # Roboto also for serif text for consistency, despite it being sans-serif
  serif: "Roboto"
  # Roboto Mono for monospace text, closely matching GitHub's code view aesthetic
  mono: "Roboto Mono"
---


# How to Efficiently Send Images to GPU 📸 🚀

Made with 💜 by Francesco Saverio Zuppichini

---
layout: default
---

# Me 
Senior Full Stack Machine Learning Engineer @Zurich Insurance

<div class="flex flex-col gap-1 items-center justify-center mt-20">
  <img class="h-40 rounded" src="assets/me.jpg"/>
  <div class="flex gap-8 items-center justify-center mt-4">
    <img class="h-12 rounded bg-white p-2" src="assets/pwc.png"/>
    <img class="h-12 rounded bg-white p-2" src="assets/v7.png"/>
    <img class="h-12 rounded bg-white p-2" src="assets/hf.png"/>
    <img class="h-12 rounded bg-white p-2" src="assets/roboflow.jpeg"/>
    <img class="h-12 rounded bg-white p-2" src="assets/zurich.png"/>
  </div>
</div>

<a href="https://www.linkedin.com/in/francesco-saverio-zuppichini-94659a150/">Find me on LI</a>

<!-- Footer -->
[^1]:[Learn More](https://sli.dev/guide/syntax.html#line-highlighting)

---
layout: default
---
# Table of contents

<Toc minDepth="1" maxDepth="1"></Toc>

---
layout: default
---
# Goals

What are we going to learn?
- 🚧 **Bottlenecks** - Finding bottleneck in our pipeline
- 📊 **Benchmark** - Let's review how to benchmark PyTorch
- 💾 **Memory Optimization** - What happens when we load images into RAM? 
- 🔢 **Data Type** - Which data type should I use?
- 🛠️ **Augmentations** - CPU, GPU or both?

<small> emoji made by our lord GPT-4</small>
---
layout: image-right
image: https://source.unsplash.com/collection/94734566/1920x1080
---

---
layout: default
---
 
# Data Type

Which data type? `uint8` or `float16`, which image quality?
---
layout: default
---
 
## `uint8`

TODO add LI post

---
layout: default
---
## Image Quality

Let's reduce the quality

```bash
# quality between 0 and 31, good between 2-5, we are sticking with 3 
ffmpeg -i "$file" -q:v 3 "$DEST_DIR/${filename}.jpg"
```

We went from `3.5MB` to `207KB`. Can you spot the difference? 🕵️‍♂️

<div class="flex  gap-1 items-center justify-center mt-8">
  <img class="h-40 rounded" src="assets/grogu.jpg"/>
  <img class="h-40 rounded" src="assets/grogu_compressed.jpg"/>
</div>

---
layout: default
---
### Dataset

Let's get a `Dataset`

```python
class FolderDataset(Dataset):
    def __init__(self, src: Path):
        self.src = src 
        self.files = list(src.glob("*"))

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        img = read_image(str(self.files[index]))
        label = torch.tensor(1)
        return img, label

    def __len__(self) -> int:
        return len(self.files)
```

and check the throughput for `.png` and `.jpeg`

---
layout: default
---
 
# Code

Use code snippets and get the highlighting directly, and even types hover![^1]

```ts {all|5|7|7-8|10|all} 
// TwoSlash enables TypeScript hover information
// and errors in markdown code blocks
// More at https://shiki.style/packages/twoslash

import { computed, ref } from 'vue'

const count = ref(0)
const doubled = computed(() => count.value * 2)

doubled.value = 2
```
---
layout: default
---
 
# Image
bla

<div class="flex items-center justify-center mt-20">
<img class=" h-40 rounded shadow" src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Cat_August_2010-4.jpg/2560px-Cat_August_2010-4.jpg"/>
</div>