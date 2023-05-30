## Language Modeling + RNNs
* Language Modeling is the task of **predicting what word comes next**
* More formally: given a sequence of words $x^1, x^2, ... x^t$, compute the probability distribution of the next word $x^{(t+1)}$
* A system does that called language model
* You can also think of Language Model as a system that assigns probability to a piece of text
	![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302131618588.png)
* How to learn a Language Model?
	Answer(pre-Deep Learning): learn an n-gram Language Model
	**Ideal**: collect statistics about how frequent different n-grams are use these to predict next word
	**Markov assumption**: $x^{(t+1)}$depends only on the preceding $n-1$ words
	![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302131631038.png)
	Below is a specific example
	![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302131634979.png)
	**Sparsity Problem1** -> numerator will be zero and thus that probability will be zero!!!
	**solution**: smoothing (add small $\delta$  to the count for every $w \in V$ ) 
		- Not really understand,  need see it later

	**Sparsity Problem2** -> What if "students open their" never occurred in data?  Then we can't calculate probability for any w!
	**partial solution**: just condition on "opened their" instead. This is called backoff
	* **Note**: increasing n makes sparsity problems worse. Typically, we can't have n bigger than 5
	
### How to build a neural Language Model?
* A fixed window neural Language Model
![](https://cdn.jsdelivr.net/gh/ambition1994/picture@main/img/202302131831936.png)
Improvements over n-gram LM:
* No sparsity problem
* Don's need to store all observed n-gram

Remaining problems:
* Fixed window is too small
* Enlarge window enlarges W
* Window can never be  large enough!
* $x^{(1)}$ and $x^{(2)}$ are multiplied by completely different wights in W. **No symmetry** in how the inputs are processed
* We need a neural architecture that can process any length input, have more sharing of the parameters, while still be sensitive to proximity -> RNN

> Prof. Manning: words order is very import in LM(Language model), if the last word is the ',', that's a really good predictor

P.S: Word2vec is a bag of words model? that means it not deep on the order of words?

RNN Advantages:
* Can process any length input
* computation for step t can (in theory) use information from many steps back
* Model size doesn't increase for longer input context
* Same weights applied on every timestep, so there is symmetry in how inputs are processed.

RNN disadvantages:
* Recurrent computation is slow
* In practice, difficult to access information from many steps back