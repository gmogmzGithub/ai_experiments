import concurrent.futures
import subprocess

# Create a list of the commands you want to run
commands = [
    ["python", "webscrapping/constitucion_de_jalisco_web_scrapper.py"],
    ["python", "webscrapping/leyes_de_aguascalientes_web_scrapper.py"],
    ["python", "webscrapping/leyes_de_campeche_web_scrapper.py"],
    ["python", "webscrapping/leyes_de_cdmx_web_scrapper.py"],
    ["python", "webscrapping/leyes_de_estado_de_mexico_web_scrapper.py"],
    ["python", "webscrapping/leyes_de_jalisco_web_scrapper.py"],
    ["python", "webscrapping/leyes_de_nuevo_leon_web_scrapper.py"],
    ["python", "webscrapping/leyes_de_queretaro_web_scrapper.py"],
    ["python", "webscrapping/leyes_federales_vigentes_web_scrapper.py"]
]


# Define a function that takes a command and runs it
def run_command(cmd):
    subprocess.run(cmd)


# Use ThreadPoolExecutor to run the commands in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(run_command, commands)
