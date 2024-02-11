# User scheduling in 5G

## How to run

First you need to install the dependencies. It is recommended to do it in a virtual environment. To do so, make sure `virtualenv` is installed (`pip install virtualenv`) and then create a virtual env in the project folder :

```
python -m venv venv
```

Then activate the virtual env. On Linux / macOS :

```
source venv/bin/activate
```

Or on Windows :

```
.\venv\Scripts\activate
```

Then, install dependencies by running :

```
pip install -r requirements.txt
```

You can run `preprocessing_analysis.py` and `greedy_analysis.py` :

```
python preprocessing_analysis.py
```

```
python greedy_analysis.py
```
