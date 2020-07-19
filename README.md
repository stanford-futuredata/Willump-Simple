# Welcome to Willump!

Willump is a Stanford-built system and research project for maximizing the performance of feature computation in machine learning inference.  Many machine learning applications have to process raw data into numerical features before they can make predictions with a machine learning model.  Willump optimizes these applications to selectively compute those features, making them much faster--in some of our experiments, almost 5x faster:

<p align="center"><img src="https://i.ibb.co/yPKPxzB/chart-1.jpg" alt="chart-1" border="0">
</p>

At a high level, Willump works by identifying a set of high-value, low-cost features and training an approximate model on them.  Then, when predicting a data input, it first predicts with the approximate model, but only returns that prediction if it is very confident, otherwise cascading to the original model.  With some algorithmic tricks that we describe in our [explanation notebook](https://github.com/stanford-futuredata/Willump-Simple/blob/master/notebooks/explanation-notebook.ipynb) and [paper](http://petereliaskraft.net/res/willump.pdf), Willump can reliably choose features and parameters that maximize inference performance without losing accuracy.

Willump is a research prototype and is not ready for production use, but it's still a system you can use and play with!  To optimize an ML application with Willump, all you have to do is write it as a Python function following an easy-to-parse format that we describe in our [tutorial notebook](https://github.com/stanford-futuredata/Willump-Simple/blob/master/notebooks/tutorial-notebook.ipynb).  If you're interested in how Willump works, please see our [explanation notebook](https://github.com/stanford-futuredata/Willump-Simple/blob/master/notebooks/explanation-notebook.ipynb), or, for even more detail, our [paper](http://petereliaskraft.net/res/willump.pdf).  We've also written a [blog post](https://dawn.cs.stanford.edu/2020/02/29/willump/) to provide an introduction to the project.

If you're wondering why this repository is called Willump-Simple, it's because this is a rewrite of the [original Willump system](https://github.com/stanford-futuredata/Willump) designed to be simpler and easier. We've released it for Willump's demo at [VLDB 2020](https://vldb2020.org/).
