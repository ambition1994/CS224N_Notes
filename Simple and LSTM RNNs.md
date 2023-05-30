### Training an RNN Language Model
* Get a big corpus of text which is a sequence of words $x^{(1)},...,x^{(T)}$
* Feed into RNN-LM; compute output distribution $\hat{y}^{(t)}$ for every step $t$
	* i.e., predict probability dist of every word, given words so far
* Loss function on step $t$ is cross-entropy between predicted probability distribution $\hat{y}^{(t)}$ , and the true next word $y^{(t)}$ (one-hot for $x^{(t+1)}$) :
 $$
J^{(t)}(\theta)=C E\left(\boldsymbol{y}^{(t)}, \hat{\boldsymbol{y}}^{(t)}\right)=-\sum_{w \in V} \boldsymbol{y}_w^{(t)} \log \hat{\boldsymbol{y}}_w^{(t)}=-\log \hat{\boldsymbol{y}}_{\boldsymbol{x}_{t+1}}^{(t)}
$$
(why the last was $-\log \hat{\boldsymbol{y}}_{\boldsymbol{x}_{t+1}}^{(t)}$ ?)
* Average this to get overall loss for entire training set:
$$
J(\theta)=\frac{1}{T} \sum_{t=1}^T J^{(t)}(\theta)=\frac{1}{T} \sum_{t=1}^T-\log \hat{\boldsymbol{y}}_{\boldsymbol{x}_{t+1}}^{(t)}
$$
**Teacher forcing**
![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302151028703.png)
> "I mean, we don't actually use what the model suggested. We penalize the model for not having suggested their. But then we just go with what's actually in the corpus and ask it to predict again." (Describe the process of teacher forcing)

* However: Computing loss and gradients across entire corpus $x^{(1)},...,x^{(T)}$ is too expensive!
$$
J(\theta)=\frac{1}{T} \sum_{t=1}^T J^{(t)}(\theta)
$$
* In practice, consider $x^{(1)},...,x^{(T)}$ as a sentence (or a document)
* Stochastic Gradient Descent allows us to compute loss and gradients for small chunk of data, and update
* Compute loss $J(\theta)$ for a sentence (actually, a batch of sentences), compute gradients and update weights. Repeat.
![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302151042761.png)
Start of sequence special symbol, end of sequence special symbol
* Just like n-gram Language Model, you can use an RNN Language Model to generate text by repeated sampling. Sample output becomes next step's input.
![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302151057282.png)
### Evaluating Language Models
* The standard evaluation metric for Language Model is **perplexity**
![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302151100714.png)
* This is equal to the exponential of the cross-entropy loss $J(\theta)$:
	* ------------- TO COMPLETE THE FORMULA --------------------------
	* lower perplexity is better!
![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302271039290.png)
### Why should we care about Language Modeling?
* Language Modeling is a bench mark task that helps us measure our progress on understanding language
* Language Modeling is a subcomponent of many NLP tasks, especially those involving generating text or estimating the probability of the text:
	![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302271046648.png)
### Recap
* Language Model: A system that predicts the next word
* Recurrent Neural Network: a family of neural networks that:
	* Take sequential input of any length
	* Apply the same weight on each step
	* Can optionally produce output on each step
* Recurrent Neural Network != Language Model
* we've shown that RNNs are a great way to build a LM
* But RNNs are useful for much more
![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302271051744.png)
![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302271053205.png)
![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302271054565.png)
![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302271058179.png)
### TODO: complete gradients vanishing  problem analysis part

### Problems with Vanishing and Exploding Gradients
![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302271112304.png)
![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302271114852.png)
### Long Short-Term Memory RNNs (LSTMs)
* A type RNN proposed by Hochreiter and Schmidhuber in 1997 as a solution to the vanishing gradients problem.
	* Everyone cites that paper but really a crucial part of teh modern LSTM is from Gers et al. (2000)
* On step $t$, there is a hidden state $h^{(t)}$ and a cell state $c^{(t)}$
	* Both are vectors length n 
	* The cell stores long-term information
	* The LSTM can read, erase, and write information from the cell
		* The cell becomes conceptually rather like RAM in a computer
* The selection of which information is erased / written / read is controlled by three corresponding gates	
	* The gates are also vectors length n
	* On each timestep, each element of the gates can be open (1), closed (0), or somewhere in-between 
	* The gates are dynamic: their value is compute based on the current context
![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302271132485.png)
![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302271135016.png)
![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302271139569.png)
### How does LSTM solve vanishing gradients?
* The LSTM architecture makes it **easier** for the RNN to **preserve information over many timesteps**.
	* e.g., if the forget gates is set to 1 for a cell dimension and the input gate set to 0, then the information of that cell is preserved indefinitely.
	* In contrast, it's harder for a vanilla RNN to learn a recurrent weight matrix $W_h$ that preserves info in the hidden state
	* In practice, you get about 100 timesteps rather than about 7
* LSTM doesn't guarantee there is no vanishing / exploding gradient, but it does provide an easier way for the model to learn long-distance dependencies
![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302271150816.png)
### Is vanishing /exploding gradient just a RNN problem?
* No! It can be a problem for all neural architectures (including **feed-forward** and **converlutional**), especially **very deep** ones.
	* Due to chain rule /choice of nonlinearity function, gradient can become vanishingly small as it backpropagates
	* Thus, lower layers are learned very slowly (hard to train)
	* Solution: Lots of new deep feedforward / convolutional architectures add more direct connections (thus allowing the gradient to flow)
	
	For example:
	* Residual connections aka "ResNet"
	* Also know as skip-connections
	* The identity connection preserves information by default 
	* This make deep networks much easier to train
![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302271158379.png)	
![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302271159132.png)
* Conclusion: Though vanishing / exploding gradients are general problem, RNNs are particularly unstable due to the repeated multiplication by the same weight matrix [Bengio et al, 1994]
![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302271217883.png)
![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302271219497.png)
![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302271220650.png)
![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302271221377.png)
### Bidirectional RNNs
* Note: bidirectional RNNs are only applicable if you have access to the **entire input sequence**
	* They are **not** applicable to Language Modeling, because in LM you **only** have **left** context available.
* If you do have entire input sequence (e.g., any kind of encoding), **bidirectionality** is powerful (you should use it by default).
* For example, **BERT** (**bidirectional** Encoder Representations from Transformers) is a powerful **pretrained contextual representation** system build on **bidirectionality**.



* 对于抽象的知识点理解, 一定要亲自做实验对应验证, 比如我在抄写RNN的实现时,加深了对之前抽象公式的理解, 甚至进一步加深对LSTM几个门的理解, 另一个领悟是之前为什么在面试时写不出来LSTM的公式, 是因为自己没有理解它的本质原理, 这个和学习数学公式理论是一样的, 一定是自己完全懂了,就觉得很“显然”了, 所以一定要把知识学到本质这个点上, 一定要实践和理论结合起来搞, 没有实际经验的理论理解一定是空中楼阁,经不起挑战.
* 其实, 我觉得本质上学习最终要的是学习“范式”, 一旦真正掌握某一种“范式”, 就可以真正的做到举一反三,事半功倍的学习效率, 但是这也离不开大量的实践, 即所谓的从“感性认识上升到理性认识.”