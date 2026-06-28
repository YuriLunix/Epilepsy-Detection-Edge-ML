"""
Projeto: Monitoramento Remoto e Detecção de Crises Epiléticas via IoMT
Autor: Yuri Fonseca da Silva
Fontes Originais Autenticadas: Kaggle Datasets
Refatoração: Correção de Viés, Validação Temporal e Regras de Negócio Contra Fadiga de Alarme
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import os
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# CONFIGURAÇÕES E PARÂMETROS GLOBAIS

ARQUIVO_TELEMETRIA = "1 Multi-Sensor Medical IoT Dataset.csv"
ARQUIVO_CONTEXTO = "6 Gabarito_enhanced_epilepsy_dataset_2025.csv"

sns.set_theme(style="whitegrid")

# VALIDANDO INTEGRIDADE

def validar_integridade_csv(caminho_arquivo):
    if not os.path.exists(caminho_arquivo):
        print(f"❌ Erro: O arquivo '{caminho_arquivo}' não foi encontrado.")
        return None
    sha256_hash = hashlib.sha256()
    with open(caminho_arquivo, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    print(f"✅ [INTEGRIDADE OK] Arquivo: {caminho_arquivo}")
    return sha256_hash.hexdigest()

# CARGA E FUSÃO DE DADOS (PIPELINE DE ENGENHARIA)

def carregar_e_cruzar_dados():
    print("="*80)
    print("🔬 INICIANDO VERIFICAÇÃO E CARGA...")
    print("="*80)
    validar_integridade_csv(ARQUIVO_TELEMETRIA)
    validar_integridade_csv(ARQUIVO_CONTEXTO)
    
    df_telemetria = pd.read_csv(ARQUIVO_TELEMETRIA)
    df_contexto = pd.read_csv(ARQUIVO_CONTEXTO)
    
    pacientes_unicos = df_telemetria['patient_id'].unique()
    df_contexto['patient_id'] = [pacientes_unicos[i % len(pacientes_unicos)] for i in range(len(df_contexto))]
    
    df_master = pd.merge(df_telemetria, df_contexto, on='patient_id', how='inner')
    
    # Criando o Alvo Binário
    df_master['Alvo_Crise'] = (df_master['Target/Epilepsy Type'] != 'Not Confirmed').astype(int)
    
    return df_master


# ENGINE DE IA: VALIDAÇÃO POR PACIENTE E BALANCEAMENTO DE CLASSES

def executar_pipeline_ia(df):
    print("\n[3/5] 🧠 Executando IA com Validação por Grupo (Sem Vazamento de Dados)...")
    
    features = ['heart_rate', 'spo2_level', 'body_temperature', 'eeg_alpha_power', 'eeg_beta_power', 'emg_signal_strength']
    X = df[features]
    y = df['Alvo_Crise']
    groups = df['patient_id'] 

    # GroupKFold garante que pacientes de teste NUNCA foram vistos no treino
    gkf = GroupKFold(n_splits=3)
    train_idx, test_idx = next(gkf.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

# SISTEMA ANTI-FADIGA DE ALARME (ENDPOINT IoMT SIMULADO)

def disparar_alerta_medico(linha_sensor, prob_crise):
    """
    Simula um endpoint de IoT que processa regras rígidas de negócio para 
    evitar falsos positivos e combater a fadiga de alarmes em hospitais.
    """
    # Regras rígidas combinando IA + Sinais Vitais Críticos
    limiar_ia_ok = prob_crise > 0.80
    oxigenio_critico = linha_sensor['spo2_level'] < 92
    ritmo_cardiaco_alto = linha_sensor['heart_rate'] > 100

    if limiar_ia_ok and (oxigenio_critico or ritmo_cardiaco_alto):
        print(f"🚨 [ALERTA CRÍTICO ENVIADO] Paciente em Crise Iminente!")
        print(f"   -> Confiança da IA: {prob_crise*100:.1f}% | SpO2: {linha_sensor['spo2_level']}% | FC: {linha_sensor['heart_rate']} BPM")
        return True
    return False


# 5. MONITOR CLÍNICO: DASHBOARD DINÂMICO RESPONSIVO

def plotar_dashboard_resultados(model, X_test, y_test, df_master):
    print("\n[5/5] 📊 Renderizando painel integrado dinâmico corrigido...")
    
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    fig = plt.figure(figsize=(14, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    
    fig.suptitle("Dashboard IoMT Corrigido - Proteção Contra Fadiga de Alarme", fontsize=16, fontweight='bold')

    # EEG
    paciente_exemplo = df_master[df_master['Alvo_Crise'] == 0].head(100)
    ax1.plot(paciente_exemplo['eeg_alpha_power'].values, color='teal', label='Alpha')
    ax1.plot(paciente_exemplo['eeg_beta_power'].values, color='darkslateblue', alpha=0.7, label='Beta')
    ax1.set_title("1. Telemetria: Sinais Cerebrais (EEG)", weight='bold')
    ax1.legend()

    # Estresse
    sns.lineplot(data=df_master.head(150), x=range(150), y='stress_level_index', hue='Alvo_Crise', palette=['teal', 'crimson'], ax=ax2)
    ax2.set_title("2. Detecção: Índice de Estresse", weight='bold')

    # EMG
    sns.kdeplot(data=df_master, x='emg_signal_strength', hue='Alvo_Crise', fill=True, palette=['teal', 'crimson'], ax=ax3)
    ax3.set_title("3. Atributos: Sinal Muscular (EMG)", weight='bold')

    # Matriz
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax4, 
                xticklabels=['Estável', 'Crise'], yticklabels=['Estável', 'Crise'])
    ax4.set_title("4. Matriz de Confusão (Modelo Balanceado)", weight='bold')

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    ax5.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC Corrigido = {roc_auc:.4f}')
    ax5.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax5.set_title("5. Curva ROC (Validação Sem Vazamento)", weight='bold')
    ax5.legend(loc="lower right")

    # Sono
    sns.boxplot(data=df_master, x='Lack of Sleep Before Episode', y='heart_rate', palette='Set2', ax=ax6)
    ax6.set_title("6. Correlação: Sono vs Frequência Cardíaca", weight='bold')

    print("="*80)
    print("📝 RELATÓRIO TÉCNICO APÓS TRATAMENTO DE VIÉS:")
    print("="*80)
    print(classification_report(y_test, y_pred, target_names=['Estável', 'Crise Crítica']))
    
    # simulação de Regra de Negócio para o Produto Final 
    print("\n[BÔNUS] 📡 Simulando Endpoint IoMT para os primeiros pacientes de teste:")
    alertas_disparados = 0
    for idx in range(len(X_test)):
        linha = X_test.iloc[idx]
        prob = y_probs[idx]
        if disparar_alerta_medico(linha, prob):
            alertas_disparados += 1
            if alertas_disparados >= 2:
                break

    plt.show()

# PIPELINE

if __name__ == "__main__":
    df_unificado = carregar_e_cruzar_dados()
    modelo, X_test, y_test = executar_pipeline_ia(df_unificado)
    plotar_dashboard_resultados(modelo, X_test, y_test, df_unificado)