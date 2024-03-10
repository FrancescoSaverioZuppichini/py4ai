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
layout: default
---

# 🚧 Bottlenecks
What happens when we load images?

<div class="flex  gap-1 items-center justify-center" >
  <img class="h-80 rounded" src="assets/bottlenecks.jpeg"/>
</div>

---
layout: default
---

## Loading from Disk
What to do?

- get a fast disk (they are cheap now xD)
- avoid opening too many file pointers
- some tasks (object detections) comes with image/labels file pairs

<div class="flex  gap-1 items-center justify-center mt-8" v-click>
  <img class="h-40 rounded" src="assets/samsung_970.png"/>
</div>

<p v-click> we'll see how we can just have one file an `memmap` it u.u </p>


---
layout: default
---

## Memory
What to do?

- prob the best thing to optimize
- we want to reduce page fault
- we want to have contigous virtual memory allocation

<p v-click> WTF is a `page`, what is `virtual memory`🤔 </p>

---
layout: default
---

### Memory - Review
Let's review some system programming stuff hehehe

- prob the best thing to optimize
- we want to reduce page fault
- we want to have contigous virtual memory allocation

<p v-click> WTF is a `page`, what is `virtual memory`🤔 </p>
---
layout: default
---

### Memory
What happens when we load images?

---
layout: default
---

# 📊 Benchmark
How to benchmark in PyTorch?

---
layout: default
---
 
# 🔢 Data Type

Which data type? `uint8` or `float16`, which image quality?

---
layout: default
---
 
## `uint8`
Should I convert to `float16` before or after I send it to GPU?

```python {1|3|5|7|all} 
x = torch.rand((batch_size, 3, 640, 480) * 255).to(torch.uint8)
# better doing this, thanks Vincent 💜
x = torch.zeroes((batch_size, 3, 640, 480), dtype=torch.uint8).random_()
# then, it is better convert it and sending to gpu
x.half().cuda()
# or viceversa?
x.cuda().half()
```

<p v-click>Raise your hand for <pre class="inline-block">x.half().cuda()</pre></p>
<p v-click>Raise your hand for <pre class="inline-block">x.cuda().half()</pre></p>
<h3 v-click>🤔🤔🤔🤔🤔🤔</h3>
<h1 v-click>🤔🤔🤔🤔🤔🤔</h1>

---
layout: default
---

## `uint8`
I've asked people

<div class="flex  gap-1 items-center justify-center" >
  <img class="h-70 rounded" src="assets/linkedin_pool.png"/>
</div>


---
layout: default
---

## `uint8`
Then benchmark it

<div class="flex  gap-1 items-center justify-center" >
  <img class="h-70 rounded" src="assets/image_to_cuda_benchmark_batch_size_vs_time.png"/>
</div>

---
layout: default
---

## `uint8`
So it is faster

<p v-click>Yes but in reality we do stuff while we run the model</p>

<div class="flex  gap-1 items-center justify-center" v-click >
  <img class="h-70 rounded" src="assets/image_to_cuda_benchmark_batch_size_vs_time.png"/>
</div>
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