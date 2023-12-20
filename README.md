

Demo Quickstart
-------------
1. Make sure you have git, python-3.11.x, and poetry installed and OPENAI key setup (see below).
2. Clone this repo and change the directory to it.
3. run `poetry install` to install dependenceies.
4. run `python -m demos.space_colony.simulation`



Installing as a PyPi Package
-------------
This will be coming once development settles a little first.



Installing Dependencies
==================


Python on Windows
----
Install Python 3.11.4
* Download and Install [Python 3.11.4](https://www.python.org/downloads/release/python-3114/) (earlier Python 3 versions may also work)
* Add python to your **Path** environment variable (e.g. `C:\Python\Python311\python.exe`)

Python on OSX and Linux
---
This is generally easier and platform specific. We recommend using homebrew on OSX and your integrated manager on linux (e.g. apt-get).


Install Poetry and Dependencies
-----
* Download and install the latest [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
* Add poetry to your **Path** environment variable (e.g. `C:\Users\Username\AppData\Roaming\pypoetry\venv\Scripts`)
* Run this command in the fable-generative-agents repo: `poetry install`

Configure OpenAI
* Create a System environment variable called **OPENAI_API_KEY** and paste your API key into the value field


Install PyCharm (optional)
--------
* Download and install the latest [PyCharm Community Edition](https://www.jetbrains.com/pycharm/download/other.html)
* Open fable-generative-agents in PyCharm
* Configure interpreter: 
  - Settings > Project > Python Interpreter > Add Interpreter > Add Local Interpreter > Poetry Environment
  - Base interpreter: `C:\Users\<Username>\AppData\Roaming\pypoetry\venv\Scripts\python.exe`
  - Check Install packages from pyproject.toml
  - Poetry executable: `C:\Users\<Username>\AppData\Roaming\Python\Scripts\poetry.exe`
  - Choose 'OK'



