#!/usr/bin/env python3
"""
openCHA + Multi-LLM + BERTScore
Interface para CSV de questões abertas
com cálculo de precisão, recall e F1
"""
from openCHA.main import openCHA
from openCHA.interface.healthchat_bertscore_multillm_interface import launch_multillm_interface

def main():
    print("🔷 openCHA + Multi-LLM + BERTScore")
    print("=" * 50)

    # Cria o agente com Multi-LLM habilitado
    cha = openCHA(
        name="openCHA-MultiLLM",
        verbose=False,
        multi_llm_enable_cache=True,
        multi_llm_timeout=180,
        multi_llm_max_workers=3,
    )

    print("🌐 Pré-inicializando modelos...")
    print("-" * 50)

    # Inicializa os modelos antes de abrir a interface
    try:
        manager = cha.get_multi_llm()
        modelos_disponiveis = manager.get_available_models()
        print(f"✅ Modelos prontos: {', '.join(modelos_disponiveis)}")
        print(f"✅ Total: {len(modelos_disponiveis)} modelo(s) inicializado(s)")
    except Exception as e:
        print(f"⚠️ Erro na pré-inicialização: {e}")
        print("   Tentando continuar mesmo assim...")

    print("-" * 50)
    print()
    print("🌐 Iniciando interface BERTScore CSV + Multi-LLM...")
    print("📍 URL: http://localhost:7860")
    print("🛑 Para parar: Ctrl+C")
    print("=" * 50)
    print()
    print("💡 Como usar:")
    print("  1. Envie seu CSV com Pergunta e Resposta")
    print("  2. Selecione os modelos: ChatGPT, Gemini e DeepSeek")
    print("  3. Clique em 'Rodar Benchmark'")
    print()

    try:
        # 🚀 Chama a nova interface que implementa o BERTScore
        launch_multillm_interface(cha)
    except KeyboardInterrupt:
        print("\n👋 openCHA encerrado")

if __name__ == "__main__":
    main()
