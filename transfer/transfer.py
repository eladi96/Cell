import os

# Main execution
if __name__ == "__main__":

    cell_dir = "none"

    while cell_dir != "abort":

        cell_dir = input("Scegliere il tipo di cellule da copiare: ")

        # Ottengo la lista delle cellule da prendere
        cell_list = os.listdir(os.getcwd() + "/cellule_originali/" + cell_dir)

        for file in os.listdir(os.getcwd() + "/cellule_rete"):

            if file in cell_list:
                old = os.getcwd() + "/cellule_rete/" + file
                new = os.getcwd() + "/cellule_rete/" + cell_dir + "/" + file
                os.rename(old, new)
