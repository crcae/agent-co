import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import time
import shutil
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- CONFIGURACIÓN DE PÁGINA (UI MODERNIZADA) ---
st.set_page_config(
    page_title="Asistente SMNYL",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS PERSONALIZADOS (GOOGLE MATERIAL STYLE) ---
st.markdown("""
    <style>
        /* Fuente global */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif;
        }
        
        /* Ocultar elementos nativos */
        header[data-testid="stHeader"] {visibility: hidden; height: 0%;}
        #MainMenu {visibility: hidden; display: none;}
        footer {visibility: hidden; display: none;}
        div[data-testid="stDecoration"] {visibility: hidden; height: 0%; display: none;}
        .stDeployButton {display: none;}

        /* Chat Messages - Estilo Burbuja Moderna */
        .stChatMessage {
            background-color: transparent;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }
        
        /* Input del chat fijo abajo y estilizado */
        .stChatInput {
            border-radius: 20px !important;
        }

        /* Sidebar moderna */
        section[data-testid="stSidebar"] {
            background-color: #f8f9fa; /* Gris muy claro estilo Google */
            border-right: 1px solid #e0e0e0;
        }
        
        /* Botones primarios estilo Google Blue */
        div.stButton > button:first-child {
            background-color: #1a73e8;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 24px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            background-color: #1557b0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }

        /* Botones de alerta (Borrar) */
        div.stButton > button.delete-btn {
            background-color: #d93025;
        }

        /* Tarjetas para archivos */
        div.file-card {
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Caching & Resources ---

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_vector_store():
    embeddings = load_embeddings()
    if os.path.exists("faiss_index_store"):
        try:
            return FAISS.load_local("faiss_index_store", embeddings, allow_dangerous_deserialization=True)
        except Exception:
            return None
    return None

# --- Core Logic --- (Misma funcionalidad, sin cambios)

def save_uploaded_files(pdf_docs):
    if not os.path.exists("saved_pdfs"):
        os.makedirs("saved_pdfs")
    for pdf in pdf_docs:
        with open(os.path.join("saved_pdfs", pdf.name), "wb") as f:
            f.write(pdf.getbuffer())

def load_documents_from_folder():
    documents = []
    if not os.path.exists("saved_pdfs"):
        os.makedirs("saved_pdfs")
        return documents
    files = [f for f in os.listdir("saved_pdfs") if f.endswith('.pdf')]
    for filename in files:
        file_path = os.path.join("saved_pdfs", filename)
        try:
            pdf_reader = PdfReader(file_path)
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    doc = Document(page_content=text, metadata={"source": filename, "page": i + 1})
                    documents.append(doc)
        except Exception as e:
            st.error(f"Error leyendo {filename}: {e}")
    return documents

def get_document_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    embeddings = load_embeddings()
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index_store")
    return True

def process_documents():
    documents = load_documents_from_folder()
    if not documents:
        return False, "No hay documentos para procesar."
    chunks = get_document_chunks(documents)
    create_vector_store(chunks)
    load_vector_store.clear()
    return True, f"Procesados {len(documents)} páginas."

def delete_files(files_to_delete):
    for filename in files_to_delete:
        file_path = os.path.join("saved_pdfs", filename)
        if os.path.exists(file_path):
            os.remove(file_path)
    success, msg = process_documents()
    return success, msg

def force_reindex():
    if os.path.exists("faiss_index_store"):
        shutil.rmtree("faiss_index_store")
    return process_documents()

# --- UI Components Mejorados ---

def get_admin_panel():
    st.title("⚙️ Panel de Control")
    st.markdown("Gestiona la base de conocimiento del asistente.")
    
    password = st.text_input("🔑 Contraseña de Acceso", type="password")
    
    if password == "admin123":
        st.success("Sesión Iniciada")
        
        # Pestañas para organizar mejor
        tab1, tab2, tab3 = st.tabs(["📚 Documentos", "📥 Cargar", "🛠️ Mantenimiento"])
        
        with tab1:
            st.subheader("Archivos Activos")
            if not os.path.exists("saved_pdfs"):
                os.makedirs("saved_pdfs")
            files = [f for f in os.listdir("saved_pdfs") if f.endswith('.pdf')]
            
            if files:
                # Mostrar archivos como tarjetas limpias
                for f in files:
                    st.markdown(f"<div class='file-card'>📄 <b>{f}</b></div>", unsafe_allow_html=True)
                
                st.markdown("### Acciones")
                files_to_delete = st.multiselect("Seleccionar para eliminar", files)
                if st.button("🗑️ Eliminar y Actualizar"):
                    if files_to_delete:
                        with st.spinner("Actualizando base de datos..."):
                            success, msg = delete_files(files_to_delete)
                            if success: st.toast("✅ Archivos eliminados correctamente", icon="🗑️")
                            time.sleep(1)
                            st.rerun()
            else:
                st.info("La biblioteca está vacía.")

        with tab2:
            st.subheader("Subir Nuevo Material")
            pdf_docs = st.file_uploader("Arrastra tus PDFs aquí", accept_multiple_files=True, type="pdf")
            if st.button("🚀 Procesar Archivos"):
                if pdf_docs:
                    with st.spinner("Leyendo y aprendiendo..."):
                        save_uploaded_files(pdf_docs)
                        success, msg = process_documents()
                        if success: st.balloons()
                else:
                    st.warning("Sube un archivo primero.")

        with tab3:
            st.warning("Zona de peligro: Usa esto solo si el bot responde cosas raras.")
            if st.button("⚠️ Resetear Cerebro (Re-Indexar)"):
                with st.spinner("Reiniciando sistema..."):
                    success, msg = force_reindex()
                    if success: st.toast("Sistema reiniciado con éxito", icon="✅")

    elif password:
        st.error("Contraseña incorrecta")

def get_chat_interface():
    st.title("💬 Asistente SMNYL")
    st.markdown("Tu copiloto experto en seguros. Pregunta sobre coberturas, trámites o manuales.")
    
    new_db = load_vector_store()
    if not new_db:
        st.error("⚠️ El sistema está apagado. Contacta al Administrador para cargar los manuales.")
        return

    # Chat Logic
    if "conversation" not in st.session_state or st.session_state.conversation is None:
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
        retriever = new_db.as_retriever(search_type="mmr", search_kwargs={"k": 50, "fetch_k": 100})
        
        prompt_template = """
        Eres el Asistente Experto de Seguros Monterrey. Responde de forma clara, profesional y empática.
        Usa la siguiente información para responder. Si no sabes, dilo.
        Contexto: {context}
        Pregunta: {question}
        Respuesta:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        st.session_state.conversation = ConversationalRetrievalChain.from_llm(
            llm=model, retriever=retriever,
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer'),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt}
        )

    # Historial de Chat con Avatares personalizados
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy tu asistente experto. ¿En qué puedo apoyarte hoy?"}]

    for msg in st.session_state.messages:
        # Avatar: Usa emojis o URLs de imágenes reales si prefieres
        avatar = "🧑‍💼" if msg["role"] == "user" else "🛡️"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Escribe tu consulta..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🛡️"):
            with st.spinner("Analizando manuales..."):
                try:
                    response = st.session_state.conversation({'question': prompt})
                    answer = response['answer']
                    st.markdown(answer)
                    
                    # Fuentes limpias (Expander minimalista)
                    sources = response.get('source_documents', [])
                    if sources:
                        unique_sources = list({f"{doc.metadata['source']}-{doc.metadata['page']}": doc for doc in sources}.values())[:3]
                        with st.expander("📚 Ver fuentes consultadas"):
                            for doc in unique_sources:
                                st.caption(f"📄 **{doc.metadata['source']}** (Pág. {doc.metadata['page']})")
                                st.markdown(f"_{doc.page_content[:150]}..._")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Ocurrió un error: {e}")

# --- MAIN APP ---
def main():
    # Cargar caché al inicio
    if os.path.exists("faiss_index_store"):
        load_vector_store()
    
    # Sidebar Minimalista
    with st.sidebar:
        st.title("Navegación")
        mode = st.radio("", ["💬 Chat", "⚙️ Admin"], index=0)
        st.markdown("---")
        st.caption("v3.0 - Powered by Gemini")

    if mode == "⚙️ Admin":
        get_admin_panel()
    else:
        get_chat_interface()

if __name__ == "__main__":
    main()
