from flask import Flask, render_template, request, jsonify, Blueprint, flash, redirect, url_for, send_from_directory, Response
from flask_sock import Sock
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import uuid
from datetime import datetime
from services.vector_service import VectorService
from services.rag_service import RAGService
from services.agent_service import AgentService
from services.speech_service import SpeechService
from services.chat_service import ChatService
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, COLLECTION_NAME, AUDIO_UPLOAD_FOLDER
import psutil
from pymilvus import Collection

# 创建蓝图
front = Blueprint('front', __name__)
admin = Blueprint('admin', __name__, url_prefix='/admin')

# 初始化服务
vector_service = VectorService()
rag_service = RAGService()
agent_service = AgentService()
speech_service = SpeechService()
chat_service = ChatService()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 前台路由
@front.route('/')
def index():
    return render_template('front/index.html')

@front.route('/chat')
def chat():
    return render_template('front/chat.html')

@front.route('/speech')
def speech():
    return render_template('front/speech.html')

@front.route('/query')
def query():
    """处理查询请求"""
    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    query_text = request.args.get('query', '').strip()
    print(f"\n开始新的查询请求: {query_text}")
    
    if not query_text:
        error_msg = "查询内容不能为空"
        print(error_msg)
        return Response(
            f"data: {json.dumps({'error': error_msg}, ensure_ascii=False)}\n\n",
            mimetype='text/event-stream',
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )

    try:
        # 使用 RAG 服务处理查询
        print("调用 RAG 服务...")
        response = Response(
            rag_service.query(query_text),
            mimetype='text/event-stream',
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )
        return response
        
    except Exception as e:
        error_msg = f"查询处理失败: {str(e)}"
        print(f"错误详情: {error_msg}")
        import traceback
        print(f"错误堆栈: {traceback.format_exc()}")
        return Response(
            f"data: {json.dumps({'error': error_msg}, ensure_ascii=False)}\n\n",
            mimetype='text/event-stream',
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )

@front.route('/chat', methods=['POST'])
def chat_message():
    """处理聊天请求"""
    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    try:
        data = request.get_json()
        if not data:
            return Response(
                "data: " + json.dumps({"error": "无效的请求数据"}, ensure_ascii=False) + "\n\n",
                mimetype='text/event-stream',
                headers={
                    'Content-Type': 'text/event-stream',
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'
                }
            )

        message = data.get('message', '').strip()
        if not message:
            return Response(
                "data: " + json.dumps({"error": "消息不能为空"}, ensure_ascii=False) + "\n\n",
                mimetype='text/event-stream',
                headers={
                    'Content-Type': 'text/event-stream',
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'
                }
            )

        def generate():
            
            try:

                for chunk in chat_service.chat(message):
                    yield chunk
            except Exception as e:
                error_msg = f"处理聊天消息失败: {str(e)}"
                print(f"错误: {error_msg}")
                yield "data: " + json.dumps({"error": error_msg}, ensure_ascii=False) + "\n\n"
                yield "data: [DONE]\n\n"
        
       

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )

    except Exception as e:
        error_msg = f"处理请求失败: {str(e)}"
        print(f"错误: {error_msg}")
        return Response(
            "data: " + json.dumps({"error": error_msg}, ensure_ascii=False) + "\n\n",
            mimetype='text/event-stream',
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )

@front.route('/clear_chat_history', methods=['POST'])
def clear_chat_history():
    """清空聊天历史"""
    try:
        chat_service.clear_history()
        return jsonify({'status': 'success', 'message': '聊天历史已清空'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'清空聊天历史失败: {str(e)}'})

@front.route('/speech/process', methods=['POST'])
def process_speech():
    """处理上传的音频文件"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': '没有上传音频文件'}), 400
            
        audio_file = request.files['audio']
        if not audio_file or not allowed_file(audio_file.filename):
            return jsonify({'error': '不支持的文件格式'}), 400

        # 保存音频文件
        filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(app.config['AUDIO_UPLOAD_FOLDER'], filename)
        audio_file.save(audio_path)

        # 处理音频文件
        result = speech_service.process_audio_file(audio_path)
        
        # 删除临时文件
        os.remove(audio_path)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'处理失败: {str(e)}'}), 500

@front.route('/speech/outbound')
def outbound_call():
    script = request.args.get('script')
    if not script:
        return jsonify({'error': '缺少初始话术'}), 400
    
    try:
        def generate():
            for message in speech_service.start_outbound_call(script):
                yield f"data: {message}\n\n"
                
        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
                'Access-Control-Allow-Origin': '*',  # 允许跨域
                'Access-Control-Allow-Methods': 'GET, POST',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# WebSocket 路由
sock = Sock()

@sock.route('/ws/speech')
def speech_socket(ws):
    script = request.args.get('script')
    if not script:
        ws.send(json.dumps({'error': '缺少初始话术'}))
        return
    
    try:
        # 发送初始话术
        ws.send(json.dumps({
            'type': 'ai_response',
            'content': script
        }))
        
        # 处理音频数据
        while True:
            audio_data = ws.receive()
            if not audio_data:
                break
                
            # 转写音频
            text = speech_service.process_audio_chunk(audio_data)
            if text and text.strip():
                # 发送转写结果
                ws.send(json.dumps({
                    'type': 'transcription',
                    'content': text
                }))
                
                # 获取 AI 回复
                response = speech_service.get_ai_response(text)
                ws.send(json.dumps({
                    'type': 'ai_response',
                    'content': response
                }))
                
    except Exception as e:
        ws.send(json.dumps({
            'type': 'error',
            'content': str(e)
        }))

# 后台路由
@admin.route('/')
def index():
    return render_template('admin/index.html')

@admin.route('/agent')
def agent():
    return render_template('admin/agent.html')

@admin.route('/upload/document', methods=['GET', 'POST'])
def upload_document():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                flash('没有文件被上传', 'error')
                return redirect(url_for('admin.upload_document'))
                
            file = request.files['file']
            if file.filename == '':
                flash('没有选择文件', 'error')
                return redirect(url_for('admin.upload_document'))
                
            if file and file.filename.lower().endswith('.pdf'):
                # 确保上传目录存在
                upload_dir = os.path.join(os.path.dirname(__file__), 'uploads/documents')
                os.makedirs(upload_dir, exist_ok=True)
                
                # 生成安全的文件名
                filename = secure_filename(file.filename)
                file_path = os.path.join(upload_dir, filename)
                
                # 保存文件
                file.save(file_path)
                
                # 处理 PDF 文件
                from services.rag_service import RAGService
                rag_service = RAGService()
                result = rag_service.process_pdf(file_path)
                flash('PDF 文件处理成功', 'success')
                return redirect(url_for('admin.index'))
            else:
                flash('只支持 PDF 文件格式', 'error')
                return redirect(url_for('admin.upload_document'))
                    
        except Exception as e:
            print(f"文件上传处理失败: {str(e)}")
            flash(f'文件处理失败: {str(e)}', 'error')
            return redirect(url_for('admin.upload_document'))
            
    # GET 请求显示上传页面
    return render_template('admin/upload_document.html')

@admin.route('/upload/audio', methods=['GET', 'POST'])
def upload_audio():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                flash('没有文件被上传', 'error')
                return redirect(url_for('admin.upload_audio'))
                
            file = request.files['file']
            if file.filename == '':
                flash('没有选择文件', 'error')
                return redirect(url_for('admin.upload_audio'))
                
            if file and any(file.filename.lower().endswith(ext) for ext in ['.mp3', '.wav', '.m4a']):
                # 确保上传目录存在
                upload_dir = os.path.join(os.path.dirname(__file__), 'uploads/audio')
                os.makedirs(upload_dir, exist_ok=True)
                
                # 生成安全的文件名
                filename = secure_filename(file.filename)
                file_path = os.path.join(upload_dir, filename)
                
                # 保存文件
                file.save(file_path)
                
                # 处理音频文件
                from services.speech_service import SpeechService
                speech_service = SpeechService()
                result = speech_service.process_audio(file_path)
                flash('音频文件处理成功', 'success')
                return redirect(url_for('admin.index'))
            else:
                flash('只支持 MP3、WAV、M4A 格式的音频文件', 'error')
                return redirect(url_for('admin.upload_audio'))
                    
        except Exception as e:
            print(f"文件上传处理失败: {str(e)}")
            flash(f'文件处理失败: {str(e)}', 'error')
            return redirect(url_for('admin.upload_audio'))
            
    # GET 请求显示上传页面
    return render_template('admin/upload_audio.html')

@admin.route('/documents')
def list_documents():
    # TODO: 实现文档列表功能
    return jsonify([])

@admin.route('/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    # TODO: 实现文档删除功能
    return jsonify({'message': '文档已删除'})

@admin.route('/agents', methods=['GET', 'POST'])
def manage_agents():
    if request.method == 'GET':
        return jsonify(agent_service.list_agents())
    else:
        data = request.get_json()
        agent = agent_service.create_agent(
            name=data['name'],
            description=data['description'],
            prompt=data['prompt'],
            functions=data['functions']
        )
        return jsonify(agent)

@admin.route('/agents/<agent_id>', methods=['GET', 'PUT', 'DELETE'])
def manage_agent(agent_id):
    if request.method == 'GET':
        agent = agent_service.get_agent(agent_id)
        if not agent:
            return jsonify({'error': 'Agent not found'}), 404
        return jsonify(agent)
    elif request.method == 'PUT':
        data = request.get_json()
        agent = agent_service.update_agent(
            agent_id=agent_id,
            name=data['name'],
            description=data['description'],
            prompt=data['prompt'],
            functions=data['functions']
        )
        if not agent:
            return jsonify({'error': 'Agent not found'}), 404
        return jsonify(agent)
    else:
        if agent_service.delete_agent(agent_id):
            return jsonify({'message': 'Agent deleted'})
        return jsonify({'error': 'Agent not found'}), 404

@admin.route('/functions')
def list_functions():
    # TODO: 实现函数列表功能
    return jsonify([
        {
            'id': 'chat',
            'name': '聊天',
            'description': '普通对话聊天'
        },
        {
            'id': 'query',
            'name': '文档检索',
            'description': '从文档中检索相关信息'
        }
    ])

@admin.route('/audio')
def list_audio():
    """列出所有音频文件"""
    audio_dir = os.path.join(os.path.dirname(__file__), 'uploads/audio')
    audio_files = []
    
    if os.path.exists(audio_dir):
        for filename in os.listdir(audio_dir):
            if filename.lower().endswith(('.mp3', '.wav', '.m4a')):
                file_path = os.path.join(audio_dir, filename)
                audio_files.append({
                    'name': filename,
                    'size': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path))
                })
    
    return render_template('admin/list_audio.html', audio_files=audio_files)

@admin.route('/audio/delete/<filename>', methods=['POST'])
def delete_audio(filename):
    """删除音频文件"""
    try:
        audio_dir = os.path.join(os.path.dirname(__file__), 'uploads/audio')
        file_path = os.path.join(audio_dir, secure_filename(filename))
        
        if os.path.exists(file_path):
            os.remove(file_path)
            flash('音频文件删除成功', 'success')
        else:
            flash('文件不存在', 'error')
            
    except Exception as e:
        flash(f'删除文件失败: {str(e)}', 'error')
    
    return redirect(url_for('admin.list_audio'))

@admin.route('/uploads/audio/<filename>')
def serve_audio(filename):
    """提供音频文件"""
    audio_dir = os.path.join(os.path.dirname(__file__), 'uploads/audio')
    return send_from_directory(audio_dir, filename)

@admin.route('/system/status')
def system_status():
    """显示系统状态"""
    try:
        collection = Collection(COLLECTION_NAME)
        milvus_status = 'Connected'
    except Exception as e:
        milvus_status = f'Not Connected: {str(e)}'
    
    status = {
        'cpu_percent': psutil.cpu_percent(),
        'memory': psutil.virtual_memory()._asdict(),
        'disk': psutil.disk_usage('/')._asdict(),
        'milvus_status': milvus_status,
        'upload_stats': {
            'documents': len(os.listdir(os.path.join(os.path.dirname(__file__), 'uploads/documents'))) if os.path.exists(os.path.join(os.path.dirname(__file__), 'uploads/documents')) else 0,
            'audio': len(os.listdir(os.path.join(os.path.dirname(__file__), 'uploads/audio'))) if os.path.exists(os.path.join(os.path.dirname(__file__), 'uploads/audio')) else 0
        }
    }
    
    return render_template('admin/system_status.html', status=status)

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config['SECRET_KEY'] = os.urandom(24)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['AUDIO_UPLOAD_FOLDER'] = os.path.join(UPLOAD_FOLDER, 'audio')
    
    # 确保上传目录存在
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(UPLOAD_FOLDER, 'audio'), exist_ok=True)
    
    # 注册蓝图
    app.register_blueprint(front)
    app.register_blueprint(admin, url_prefix='/admin')
    
    # 注册 WebSocket
    sock.init_app(app)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
