# Importações constantes

import pandas as pd
import re
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from optuna import create_study, Trial


# Coisas definidas previamente

# num de features
num_dados_de_entrada = 69

# num de targets
num_dados_de_saida = 1

# quantas vezes iremos rodar o optuna
NUM_TENTATIVAS = 110

# 10% dos dados gerais p/ separados para teste
TAMANHO_TESTE = 0.1

# 10% dos dados do treino p/ validar 
TAMANHO_VALIDACAO = 0.1

# Semente aleatória petista
SEMENTE_ALEATORIA = 13

NUM_EPOCAS = 60


# ## DataModule (completo)

class DataModule(L.LightningDataModule):
    def __init__(
        self,
        tamanho_lote=256,
        num_trabalhadores=2,
    ):
        super().__init__()

        self.tamanho_lote = tamanho_lote
        self.num_trabalhadores = num_trabalhadores

    def prepare_data(self):

        pd.read_csv("dataset_tratado.csv")

    def setup(self, stage):

        features = ["Thermodynamic stability level", "Energy", "Work function", "Heat of formation", "Space group number", "Volume of unit cell", "Electronegativity", "Be", "As", "O", "Ca", "Fe", "S", "In","Se","Sc","V","Zr","B","H","Te","Al","Mg","Ba","Pb","Mn","Si","Cr","Br","Ga","Hf","Ge","Ti","C","I","Li","Cl","Sr","Na","Nb","Ni","Ta","Pd","Pt","Tl","W","Sb","N","Cd","Cu","Sn","F","P","Ag","Au","Bi","Co","Zn","Rb","Os","Hg","Ir","Mo","Re","Rh","Ru","Y","Cs","K"]
        target = ["Band gap"]

        df = pd.read_csv("dataset_tratado.csv")
        
        df = df.reindex(features + target, axis=1)
        df = df.dropna()
        
        indices = df.index
        indices_treino_val, indices_teste = train_test_split(
            indices, test_size=TAMANHO_TESTE, random_state=SEMENTE_ALEATORIA
        )

        df_treino_val = df.loc[indices_treino_val]
        df_teste = df.loc[indices_teste]

        indices = df_treino_val.index
        indices_treino, indices_val = train_test_split(
            indices,
            test_size=TAMANHO_TESTE,
            random_state=SEMENTE_ALEATORIA,
        )

        df_treino = df.loc[indices_treino]
        df_val = df.loc[indices_val]
        
        X_treino = df_treino.reindex(features, axis=1).values
        y_treino = df_treino.reindex(target, axis=1).values

        self.x_scaler = MaxAbsScaler()
        self.x_scaler.fit(X_treino)

        self.y_scaler = MaxAbsScaler()
        self.y_scaler.fit(y_treino)

        if stage == "fit":
            X_val = df_val.reindex(features, axis=1).values
            y_val = df_val.reindex(target, axis=1).values

            X_treino = self.x_scaler.transform(X_treino)
            y_treino = self.y_scaler.transform(y_treino)

            X_val = self.x_scaler.transform(X_val)
            y_val = self.y_scaler.transform(y_val)

            self.X_treino = torch.tensor(X_treino, dtype=torch.float32)
            self.y_treino = torch.tensor(y_treino, dtype=torch.float32)

            self.X_val = torch.tensor(X_val, dtype=torch.float32)
            self.y_val = torch.tensor(y_val, dtype=torch.float32)

        if stage == "test":
            X_teste = df_teste.reindex(features, axis=1).values
            y_teste = df_teste.reindex(target, axis=1).values

            X_teste = self.x_scaler.transform(X_teste)
            y_teste = self.y_scaler.transform(y_teste)

            self.X_teste = torch.tensor(X_teste, dtype=torch.float32)
            self.y_teste = torch.tensor(y_teste, dtype=torch.float32)

    def train_dataloader(self):
        return DataLoader(
            TensorDataset(self.X_treino, self.y_treino),
            batch_size=self.tamanho_lote,
            num_workers=self.num_trabalhadores,
        )

    def val_dataloader(self):
        return DataLoader(
            TensorDataset(self.X_val, self.y_val),
            batch_size=self.tamanho_lote,
            num_workers=self.num_trabalhadores,
        )

    def test_dataloader(self):
        return DataLoader(
            TensorDataset(self.X_teste, self.y_teste),
            batch_size=self.tamanho_lote,
            num_workers=self.num_trabalhadores,
        )


# ## Criando a rede neural com PyTorch Lightning

class MLP(L.LightningModule):
    def __init__(
        self, num_camadas, num_neuronios, funcao_de_ativacao, otimizador, taxa_de_aprendizado, num_dados_entrada, num_targets
    ):
    
    # def __init__(
    #     self, num_dados_entrada, neuronios_c1, neuronios_c2, num_targets
    # ):
        super().__init__()
        
        camadas = []
        
        camadas.append(nn.Linear(num_dados_entrada, num_neuronios))
        camadas.append(funcao_de_ativacao)
        
        for _ in range(num_camadas-1):
            camadas.append(nn.Linear(num_neuronios, num_neuronios))
            camadas.append(funcao_de_ativacao)
            
        camadas.append(nn.Linear(num_neuronios, num_targets))
        
        
        self.todas_camadas = nn.Sequential(*camadas)
        
        self.otimizador = otimizador
        self.taxa_de_aprendizado = taxa_de_aprendizado

        # self.camadas = nn.Sequential(
        #     nn.Linear(num_dados_entrada, neuronios_c1),
        #     nn.Sigmoid(),
        #     nn.Linear(neuronios_c1, neuronios_c2),
        #     nn.Sigmoid(),
        #     nn.Linear(neuronios_c2, num_targets),
        # )

        self.fun_perda = F.mse_loss

        self.perdas_treino = []
        self.perdas_val = []

        self.curva_aprendizado_treino = []
        self.curva_aprendizado_val = []

    def forward(self, x):
        x = self.todas_camadas(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.fun_perda(y, y_pred)

        self.log("loss", loss, prog_bar=True)
        self.perdas_treino.append(loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.fun_perda(y, y_pred)

        self.log("val_loss", loss, prog_bar=True)
        self.perdas_val.append(loss)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.fun_perda(y, y_pred)

        self.log("test_loss", loss)

        return loss

    def on_train_epoch_end(self):
        # Atualiza curva de aprendizado
        perda_media = torch.stack(self.perdas_treino).mean()
        self.curva_aprendizado_treino.append(float(perda_media))
        self.perdas_treino.clear()

    def on_validation_epoch_end(self):
        # Atualiza curva de aprendizado
        perda_media = torch.stack(self.perdas_val).mean()
        self.curva_aprendizado_val.append(float(perda_media))
        self.perdas_val.clear()

    def configure_optimizers(self):
        if self.otimizador == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=self.taxa_de_aprendizado)
        elif self.otimizador == "ADAM":
            optimizer = optim.Adam(self.parameters(), lr=self.taxa_de_aprendizado)
        elif self.otimizador == "RMSPROP":
            optimizer = optim.RMSprop(self.parameters(), lr=self.taxa_de_aprendizado)
        # else:
#             fazer erro
            
        return optimizer



# ## Otimização de hipercondríacos  (Optuna)

def cria_instancia_modelo(trial):
    
    todas_funcoes_ativacao = {
        "Sigmoide" : nn.Sigmoid(),
        "Relu" : nn.ReLU(),
        "Tangente_Hiperbólica" : nn.Tanh(),
        "ELU" : nn.ELU(),
        "LeakyReLU" : nn.LeakyReLU()
    }

    num_neuronios = trial.suggest_int("num_neuronios", 30, 1000)  
    num_camadas = trial.suggest_int("num_camadas", 3, 10)
    funcao_de_ativacao = trial.suggest_categorical("funcao_de_ativacao", ["Sigmoide", "Relu", "Tangente_Hiperbólica", "ELU", "LeakyReLU"])

    taxa_de_aprendizado = trial.suggest_float("taxa_de_aprendizado", 0.000001, 0.5, log=True)
    
    otimizador = trial.suggest_categorical("otimizador", ["SGD", "ADAM", "RMSPROP"])
    
    # pegando a função paitonica da função de ativação dada pelo sugest de "funcao_de_ativacao"
    real_funcao_ativacao = todas_funcoes_ativacao[funcao_de_ativacao]
            

    model = MLP(num_camadas = num_camadas, num_neuronios = num_neuronios, funcao_de_ativacao = real_funcao_ativacao, otimizador = otimizador, taxa_de_aprendizado = taxa_de_aprendizado, num_dados_entrada = num_dados_de_entrada, num_targets = num_dados_de_saida)

    return model


# Criando a função objetivo

def funcao_objetivo(
    trial
):
    modelo = cria_instancia_modelo(trial)
    
#     P/ rodar no computador normal:
    # treinador = L.Trainer(max_epochs=NUM_EPOCAS)
    
#     P/ GPU:
    treinador = L.Trainer(max_epochs=NUM_EPOCAS, accelerator="gpu")
    
    dm = DataModule()
    
    treinador.fit(modelo, dm)
    
    modelo.eval()
    dm.setup("test")

    with torch.no_grad():
        X_true = dm.X_teste

        y_true = dm.y_teste
        y_true = dm.y_scaler.inverse_transform(y_true)

        y_pred = modelo(X_true)
        y_pred = dm.y_scaler.inverse_transform(y_pred)

        RMSE = mean_squared_error(y_true, y_pred, squared=False)

    return RMSE


# criando obj de estudo

objeto_de_estudo = create_study(direction="minimize")


objeto_de_estudo.optimize(funcao_objetivo, n_trials=NUM_TENTATIVAS)

# observando os resultados

df = objeto_de_estudo.trials_dataframe()

print(df)

# melhor trial achando pelo optuna, ou seja, os melhores hiperparametros achados para a sua rede

melhor_trial = objeto_de_estudo.best_trial


print(melhor_trial)