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

# --- Caching & Resources ---

@st.cache_resource
def load_embeddings():
    """
    Loads and caches the HuggingFace Embeddings model.
    This prevents reloading the model (which is heavy) on every interaction.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_vector_store():
    """
    Loads and caches the FAISS Vector Store from disk.
    This prevents reading from the hard drive on every interaction.
    """
    embeddings = load_embeddings()
    if os.path.exists("faiss_index_store"):
        try:
            return FAISS.load_local("faiss_index_store", embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            return None
    return None

# --- Core Logic: Source of Truth & Ingestion ---

def save_uploaded_files(pdf_docs):
    """
    Saves uploaded PDF files to 'saved_pdfs' directory.
    """
    if not os.path.exists("saved_pdfs"):
        os.makedirs("saved_pdfs")
    
    for pdf in pdf_docs:
        with open(os.path.join("saved_pdfs", pdf.name), "wb") as f:
            f.write(pdf.getbuffer())

def load_documents_from_folder():
    """
    Scans 'saved_pdfs' folder, reads all PDFs, and returns a list of Document objects with metadata.
    This is the 'Source of Truth' logic.
    """
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
                    # Create a Document object with text and metadata (source filename and page number)
                    doc = Document(
                        page_content=text,
                        metadata={"source": filename, "page": i + 1}
                    )
                    documents.append(doc)
        except Exception as e:
            st.error(f"Error al leer {filename}: {e}")
            
    return documents

def get_document_chunks(documents):
    """
    Splits documents into chunks while preserving metadata.
    """
    # Increased chunk size to keep tables and logical sections together
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    """
    Creates and saves a FAISS vector store from document chunks using local HuggingFace embeddings.
    Saves to 'faiss_index_store' folder.
    """
    # Use cached embeddings
    embeddings = load_embeddings()
    
    # Create vector store from documents (preserves metadata)
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    
    # Save locally
    vector_store.save_local("faiss_index_store")
    return True

def process_documents():
    """
    Orchestrates the full ingestion pipeline: Load from Folder -> Split -> Embed -> Save Index.
    """
    documents = load_documents_from_folder()
    if not documents:
        return False, "No hay documentos en la carpeta 'saved_pdfs'."
    
    chunks = get_document_chunks(documents)
    create_vector_store(chunks)
    
    # Clear the cache so the new index is loaded next time
    load_vector_store.clear()
    
    return True, f"Procesados {len(documents)} páginas de {len(set(d.metadata['source'] for d in documents))} archivos."

def delete_files(files_to_delete):
    """
    Deletes selected files from 'saved_pdfs' and triggers re-indexing.
    """
    for filename in files_to_delete:
        file_path = os.path.join("saved_pdfs", filename)
        if os.path.exists(file_path):
            os.remove(file_path)
    
    # Re-index immediately to reflect changes
    success, msg = process_documents()
    return success, msg

def force_reindex():
    """
    Deletes the existing index and rebuilds it from scratch using all files in 'saved_pdfs'.
    """
    if os.path.exists("faiss_index_store"):
        shutil.rmtree("faiss_index_store")
    
    return process_documents()

# --- UI Components ---

def get_admin_panel():
    """
    Renders the Admin Panel for document ingestion and management.
    """
    st.header("🔒 Panel de Administrador")
    
    password = st.text_input("Ingrese Contraseña de Administrador", type="password")
    
    if password == "admin123":
        st.success("Acceso Concedido")
        
        st.subheader("Gestión de Base de Conocimiento")
        
        # 1. List and Manage Files
        st.markdown("### 📂 Archivos en la Base de Conocimiento")
        
        if not os.path.exists("saved_pdfs"):
            os.makedirs("saved_pdfs")
            
        files = [f for f in os.listdir("saved_pdfs") if f.endswith('.pdf')]
        
        if files:
            # Display files
            st.dataframe({"Nombre del Archivo": files}, use_container_width=True)
            
            # Delete Interface
            st.markdown("#### 🗑️ Eliminar Archivos")
            files_to_delete = st.multiselect("Seleccionar archivos para eliminar", files)
            
            if st.button("🗑️ Eliminar Seleccionados y Actualizar"):
                if files_to_delete:
                    with st.spinner("Eliminando archivos y actualizando cerebro..."):
                        success, msg = delete_files(files_to_delete)
                        if success:
                            st.success("✅ Archivos eliminados y cerebro actualizado.")
                        else:
                            st.warning(msg) # Might happen if all files are deleted
                        time.sleep(2)
                        st.rerun()
                else:
                    st.warning("Seleccione al menos un archivo para eliminar.")
        else:
            st.info("No hay archivos guardados aún.")
            
        st.markdown("---")
        
        # 2. Upload New Files
        st.markdown("### 📤 Cargar Nuevos Manuales")
        pdf_docs = st.file_uploader("Seleccionar PDFs", accept_multiple_files=True)
        
        if st.button("Cargar y Actualizar Base de Conocimiento"):
            if not pdf_docs:
                st.warning("Por favor, sube al menos un archivo PDF.")
            else:
                with st.spinner("Guardando archivos y procesando TODOS los documentos..."):
                    # Save new files
                    save_uploaded_files(pdf_docs)
                    
                    # Trigger full re-process
                    success, msg = process_documents()
                    
                    if success:
                        st.success(f"✅ ¡Éxito! {msg}")
                        st.info("La base de conocimiento está sincronizada con la carpeta.")
                    else:
                        st.error(f"Error: {msg}")
                        
                    time.sleep(2)
                    st.rerun()

        st.markdown("---")
        st.markdown("### 🔍 Probador de Búsqueda (Debug)")
        st.info("Escribe un término para ver qué fragmentos exactos recupera el cerebro.")
        test_query = st.text_input("Prueba qué está viendo el cerebro (ej. 'Mucopolisacaridosis')")
        if test_query:
            # Use cached vector store
            new_db = load_vector_store()
            if new_db:
                try:
                    # Search with k=5
                    docs = new_db.similarity_search(test_query, k=5)
                    
                    st.write(f"Resultados para: **'{test_query}'**")
                    for i, doc in enumerate(docs):
                        with st.expander(f"Resultado {i+1}: {doc.metadata.get('source', 'Desconocido')} (Pág. {doc.metadata.get('page', 'N/A')})"):
                            st.text(doc.page_content)
                except Exception as e:
                    st.error(f"Error al buscar: {e}")
            else:
                st.warning("No hay base de datos para buscar.")

        st.markdown("---")
        st.markdown("### ⚠️ Zona de Peligro")
        if st.button("⚠️ Forzar Re-Indexación Completa"):
            with st.spinner("Borrando base de datos y reconstruyendo desde cero..."):
                success, msg = force_reindex()
                if success:
                    st.success(f"✅ Base de datos reconstruida: {msg}")
                else:
                    st.error(f"Error: {msg}")
                time.sleep(2)
                st.rerun()

    elif password:
        st.error("Contraseña incorrecta")

def get_chat_interface():
    """
    Renders the Chat Interface for Advisors.
    """
    st.header("💬 Asistente para Asesores")
    
    # Use cached vector store
    new_db = load_vector_store()
    
    if not new_db:
        st.warning("⚠️ Sistema no inicializado. Por favor pida al Administrador que cargue los documentos en el Panel de Admin.")
        return

    # Optimization: Use MMR with aggressive retrieval (k=50) for Gemini 2.0 Flash context window
    retriever = new_db.as_retriever(search_type="mmr", search_kwargs={"k": 50, "fetch_k": 100})

    # Initialize Chat Logic
    if "conversation" not in st.session_state or st.session_state.conversation is None:
         # Use gemini-2.0-flash as requested (Working version)
         model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
         
         prompt_template = """
         Eres el Asistente IA Experto de Seguros Monterrey New York Life. 
         Tienes acceso a una biblioteca de documentos (manuales, condiciones generales, etc.).
         Tu función es apoyar a los asesores internos con información precisa.

         Instrucciones CRÍTICAS:
         1. Al responder, consulta la información disponible en el contexto.
         2. Si encuentras la respuesta, cita explícitamente el nombre del documento fuente.
         3. ANTES de responder afirmativamente sobre una cobertura, DEBES verificar la sección 'VII. Exclusiones Generales'.
         4. Regla Crítica: Si la actividad involucra 'carreras', 'pruebas de velocidad' o 'concursos' en vehículos de cualquier tipo, está EXCLUIDA.
         5. Si encuentras una contradicción entre Cobertura y Exclusión, la Exclusión SIEMPRE gana.
         6. Cuando te pregunten sobre rankings, estadísticas o tablas (como 'casos más costosos' o 'descuentos'), analiza cuidadosamente los fragmentos de texto sin formato, ya que el formato de tabla puede haberse perdido. Reconstruye las filas de datos para encontrar la respuesta.
         7. Si la información NO está en el contexto, responde: "No encuentro esa información en los documentos cargados actualmente".

         Contexto:
         {context}

         Pregunta:
         {question}

         Respuesta:
         """
         prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
         
         st.session_state.conversation = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=retriever,
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer'),
            return_source_documents=True, # Enable source retrieval for citation
            combine_docs_chain_kwargs={"prompt": prompt}
        )

    # Display Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle User Input
    if prompt := st.chat_input("Escribe tu pregunta sobre los manuales..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analizando biblioteca de documentos..."):
                max_retries = 3
                retry_delay = 10
                
                for attempt in range(max_retries):
                    try:
                        response = st.session_state.conversation({'question': prompt})
                        answer = response['answer']
                        st.markdown(answer)
                        
                        # Professional Source Citation
                        sources = response.get('source_documents', [])
                        if sources:
                            # Deduplicate sources based on page number AND filename
                            seen_sources = set()
                            unique_sources = []
                            for doc in sources:
                                page = doc.metadata.get('page', 'N/A')
                                source_file = doc.metadata.get('source', 'Desconocido')
                                identifier = f"{source_file}-{page}"
                                
                                if identifier not in seen_sources:
                                    seen_sources.add(identifier)
                                    unique_sources.append(doc)
                            
                            # Display sources in expanders
                            for doc in unique_sources[:3]: # Show top 3 unique sources
                                page_num = doc.metadata.get('page', 'Desconocida')
                                source_file = doc.metadata.get('source', 'Documento')
                                with st.expander(f"📚 Fuente: {source_file} (Pág. {page_num})"):
                                    st.markdown(f"**Extracto:**")
                                    st.text(doc.page_content[:500] + "...") 

                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        break # Success, exit loop
                    except Exception as e:
                        error_msg = str(e)
                        if "429" in error_msg or "ResourceExhausted" in error_msg or "quota" in error_msg.lower():
                            if attempt < max_retries - 1:
                                st.warning(f"⚠️ Límite de cuota alcanzado. Reintentando en {retry_delay} segundos... (Intento {attempt + 1}/{max_retries})")
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                            else:
                                st.error("❌ Se ha excedido la cuota de la API de Google. Por favor intenta de nuevo en unos minutos.")
                        else:
                            st.error(f"Error al generar respuesta: {e}")
                            break

def main():
    st.set_page_config(page_title="Asistente de Seguros", page_icon="🛡️", layout="wide")
    
# --- ESTRATEGIA PARCHE PARA TAPAR MARCA DE AGUA ---
    hide_streamlit_style = """
                <style>
                /* 1. Ocultar elementos estándar */
                #MainMenu {visibility: hidden;}
                header {visibility: hidden;}
                footer {visibility: hidden;}
                
                /* 2. EL PARCHE: Un recuadro blanco fijo en la esquina inferior derecha */
                div[data-testid="stAppViewContainer"]::after {
                    content: "";
                    position: fixed;
                    bottom: 0;
                    right: 0;
                    width: 200px;  /* Ancho suficiente para tapar el botón */
                    height: 40px;  /* Alto suficiente para tapar el botón */
                    background-color: white; /* Color del fondo de tu app */
                    z-index: 999999; /* Prioridad máxima para estar encima */
                    pointer-events: none; /* Para no bloquear clics si te pasas de tamaño */
                }
                
                /* 3. Intento directo de ocultar la clase del visor (a veces funciona) */
                .viewerBadge_container__1QSob {
                    display: none !important;
                }
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    # -------------------------------------------------------

    # Startup Cache Warming
    # Try to load the vector store immediately so it's ready for the user
    if os.path.exists("faiss_index_store"):
        load_vector_store()
    
    # Sidebar Navigation
    st.sidebar.title("Navegación")
    mode = st.sidebar.radio("Seleccione Modo:", ["💬 Chat Asistente", "🔒 Panel Admin"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("Sistema RAG Multi-PDF\nv2.2 (Optimized)")

    if mode == "🔒 Panel Admin":
        get_admin_panel()
    else:
        get_chat_interface()

if __name__ == "__main__":
    main()
