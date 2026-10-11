# Installation Guide (Windows 11)

**All commands in this guide are run in Windows PowerShell.** Open PowerShell from the Start menu and enter the commands shown below.

## 1. Install Git

If Git is not already installed, download and install **Git for Windows**:

https://git-scm.com/download/win

Open PowerShell and verify the installation:

```powershell
git --version
```

This guide assumes Python is already installed.

## 2. Clone the repository

In PowerShell, navigate to the directory where you want to keep the project. For example, to go to your Desktop:

```powershell
cd "$HOME\Desktop"
```

Clone the repository:

```powershell
git clone https://github.com/Natique1415/IPL-Winner-Project.git
```

Move into the project directory:

```powershell
cd .\IPL-Winner-Project
```

Make sure `matches.csv`, `deliveries.csv`, `requirements.txt`, and the project folders are present in the project root. If you have `deliveries.csv.zip`, extract `deliveries.csv` into the project root so it sits beside `matches.csv`.

## 3. Create and activate a virtual environment

Create the environment:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this for the current session and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## 4. Install dependencies

From the project root, with the virtual environment activated, run:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. Run in this order

1. `viewing-data.py`
2. `team-renaming.py` (optional)
3. `cleaning-data.py`
4. `decision-tree.py`
5. `nb.py`
6. `knn.py`
7. `compare-models.py`
