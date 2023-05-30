### Section1: Pre-Neural Machine Translation
Machine Translation (MT) is the task of translating a sentence x from one language (the source language) to a sentence y in another language (the target language).


### 1990s-2010s: Statistical Machine Translation
* Core idea: Learn a probabilistic model from data
* Suppose we're translating French -> English
* We want to find best English sentence $y$, given French sentence $x$
 $$
\operatorname{argmax}_y P(y \mid x)
$$
* Use Bayes Rule to break this down into two components  to be learned separately:
$$
=\operatorname{argmax}_y P(x \mid y) P(y)
$$
![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202303101143803.png)
* Question: How to learn translation model $P(x \mid y)$ ?
* First, need large amount of parallel data (e.g., pairs of human-translated French/English sentence)
* Question: How to learn translation model $P(x \mid y)$ from the parallel corpus ?
* Break it down further: Introduce latent a variable into the model: $P(x, a \mid y)$ (not understand !!!)
   where a is the alignment, i.e. word-level correspondence between source sentence $x$ and target sentence $y$

![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202303101154221.png)

### What is alignment ?
Alignment is the correspondence between particular words in the translated sentence pair.
* Typological difference between languages lead to complicated alignments !
* Note: Some words have no counterpart

### Learning alignment for SMT
* We learn $P(x, a \mid y)$ as a combination of many factors, including:
	* probability of particular words aligning (also depends on position in sent)
	* probability of particular words having a particular fertility (number of corresponding words)
	* etc.
* Alignments $a$ are latent variables: They aren't explicitly specified in the data !
	* Require the use of special learning algorithms (like Expectation-Maximization) for learning int  parameters of distributions with latent variables