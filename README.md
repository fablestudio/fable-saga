Installing
=====

Install Python 3.11.4
* Download and Install [Python 3.11.4](https://www.python.org/downloads/release/python-3114/) (earlier Python 3 versions may also work)
* Add python to your **Path** environment variable (e.g. `C:\Python\Python311\python.exe`)

Install Poetry
* Download and install the latest [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
* Add poetry to your **Path** environment variable (e.g. `C:\Users\Username\AppData\Roaming\pypoetry\venv\Scripts`)
* Run this command in the fable-generative-agents repo: `poetry install`

Configure OpenAI
* Create an environment variable called **OPENAI_API_KEY** and paste your API key into the value field
  
Install PyCharm
* Download and install the latest [PyCharm Community Edition](https://www.jetbrains.com/pycharm/download/other.html)
* Open fable-generative-agents in PyCharm
* Configure interpreter: 
  - Settings > Project > Python Interpreter > Add Interpreter > Add Local Interpreter > Poetry Environment
  - Base interpreter: `C:\Users\Chris\AppData\Roaming\pypoetry\venv\Scripts\python.exe`
  - Check Install packages from pyproject.toml
  - Poetry executable: `C:\Users\Username\AppData\Roaming\Python\Scripts\poetry.exe`
  - Choose 'OK'



