# Set PATH to include Poppler bin directory
os.environ['PATH'] += os.pathsep + r'C:\Users\Frank McClane\PycharmProjects\PDF_MCBC\.venv\Lib\site-packages\poppler-24.02.0\Library\bin'

# Verify Poppler installation
def verify_poppler():
    if os.system("pdftoppm -h") != 0:
        raise EnvironmentError("Poppler is not installed or not found in PATH")
