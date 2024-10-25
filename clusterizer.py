import pandas as pd
from pandas import DataFrame
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import os
import re
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from typing import List


class Clusterizer:
    def __init__(self):
        self.files_data = []
        self.files_index = []
        self.current_df = None
        self.current_df_indexes = None
        self.dfs_per_timestep = []
        self.idx_per_timestep = []
        self.species_len = 0

    def get_files(self):
        files = os.listdir('pops3')

        files_data = [file for file in files if re.search(r'Data', file)]
        files_index = [file for file in files if re.search(r'Ids', file)]

        # Ordena arquivos por timestep (numero pós o "t")
        self.files_data = sorted(files_data, key=lambda x: int(x.split('t')[1].split()[0]))
        self.files_index = sorted(files_index, key=lambda x: int(x.split('t')[1].split()[0]))

    def _load_data(self, timestep: int):
        self.current_df = pd.read_csv(f'pops3/{self.files_data[timestep]}', sep='\t', header=None)
        self.current_df.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

        self.current_df_indexes = pd.read_csv(f'pops3/{self.files_index[timestep]}', sep='\t', header=None)
        self.current_df_indexes.columns = ['ancestral', 'id']

        self.current_df['id'] = self.current_df_indexes['id']
        self.current_df['ancestor'] = self.current_df_indexes['ancestor']
    
    def _agroup_species(self, data: DataFrame):
        dbscan = DBSCAN(eps=50, min_samples=1, n_jobs=-1)

        # se não for o primeiro timestep, pega o último dataframe para pegar os clusters anteriores
        last_df = self.dfs_per_timestep[-1] if self.dfs_per_timestep != [] else None

        if last_df is not None:
            # Populações de mesmo índice receberão o mesmo cluster do timestep anterior
            data = pd.merge(data, last_df[['id', 'species', 'ancestor']], on='id', how='left')
            # as novas populações são do mesmo cluster (espécie) do ancestra
            data['species'] = data['species'].fillna(data['ancestor'])
            
        else:
            data['species'] = -1

        # agrupar por espécie
        for i in data['species'].unique():
            group = data.loc[data['species'] == i]
            group = group.drop(columns=['species', 'id'])
            species = np.array(dbscan.fit_predict(group))
            species = species + self.species_len

            # documentar: os labels não carregam info temporal ->
            # o zero desse timestep, não necesseriamente é do mesmo cluster
            # do zero do timestep anterior
            self.species_len += len(np.unique(species))

            data.loc[data['species'] == i, 'species'] = species
            
        return data

    def parse_timesteps(self):
        for i in range(len(self.files_data)):
            print(f'Parsing timestep {i}')

            self._load_data(i)
            self.current_df = self._agroup_species(self.current_df)
            self.dfs_per_timestep.append(self.current_df)

    def animate_clusters(self, save_as="clusters_animation.gif"):
        fig, ax = plt.subplots()
        
        def update_plot(timestep):
            ax.clear()

            species = self.dfs_per_timestep[timestep]['species']
            
            # Executa o TSNE para o timestep atual
            tsne = TSNE(n_components=2, random_state=42)
            data_tsne = tsne.fit_transform(self.dfs_per_timestep[timestep].drop(columns=['species', 'id']))

            # Scatter plot
            ax.scatter(data_tsne[:, 0], data_tsne[:, 1], c=species, cmap='tab10')
            ax.set_title(f'Timestep {timestep}')
        
        # Configura a animação
        ani = animation.FuncAnimation(fig, update_plot, frames=len(self.dfs_per_timestep), repeat=True)
        
        # Salvar como GIF
        ani.save(save_as, writer='imagemagick', fps=2)  # Pode salvar como mp4 também, trocando writer
        plt.show()


if __name__ == '__main__':
    clusterizer = Clusterizer()
    clusterizer.get_files()
    clusterizer.parse_timesteps()

