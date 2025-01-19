import os
import json
import time
import wave
import ssl
import requests
from typing import Optional
import websocket
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import FUNASR_API_URL, QWEN_API_URL, AUDIO_UPLOAD_FOLDER, QWEN_MODEL

class SpeechService:
    def __init__(self):
        os.makedirs(AUDIO_UPLOAD_FOLDER, exist_ok=True)
        # 配置带有重试机制的会话
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    def transcribe_audio(self, audio_path: str) -> str:
        """使用 FunASR WebSocket 转写音频文件"""
        try:
            # 读取音频文件
            with wave.open(audio_path, 'rb') as wav_file:
                audio_bytes = wav_file.readframes(wav_file.getnframes())
                sample_rate = wav_file.getframerate()
                if sample_rate != 16000:
                    raise ValueError("音频采样率必须为16kHz")

            # 配置SSL上下文
            ssl_context = ssl.SSLContext()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # 连接WebSocket服务器
            uri = "wss://127.0.0.1:10096"
            print(f"连接FunASR服务: {uri}")
            
            ws = websocket.create_connection(
                uri,
                sslopt={
                    "cert_reqs": ssl.CERT_NONE,
                    "check_hostname": False
                }
            )
            
            try:
                # 发送初始配置
                message = {
                    "mode": "offline",
                    "chunk_size": "5, 10, 5",
                    "chunk_interval": 10,
                    "wav_name": os.path.basename(audio_path),
                    "is_speaking": True
                }
                ws.send(json.dumps(message))
                print("发送配置信息:", message)

                # 发送音频数据
                ws.send_binary(audio_bytes)
                print(f"发送音频数据: {len(audio_bytes)} 字节")

                # 发送结束信号
                ws.send(json.dumps({"is_speaking": False}))
                print("发送结束信号")

                # 等待并接收结果
                result = ""
                while True:
                    try:
                        msg = ws.recv()
                        if not msg:
                            continue
                            
                        response = json.loads(msg)
                        print("收到响应:", response)
                        
                        if "text" in response:
                            result = response["text"]
                            break
                    except websocket.WebSocketConnectionClosed:
                        break
                    except json.JSONDecodeError as e:
                        print(f"JSON解析错误: {e}")
                        continue

                if not result:
                    raise ValueError("未收到有效的转写结果")

                return result

            finally:
                ws.close()

        except Exception as e:
            print(f"音频转写失败: {str(e)}")
            raise

    def summarize_text(self, text: str) -> str:
        """使用 Qwen 2.5 总结文本"""
        try:
            # 构建请求数据
            system_prompt = "请对以下文本进行简要总结，突出重点内容。"
            request_data = {
                "model": QWEN_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                "stream": True
            }
            
            # 发送请求到 Qwen API
            response = requests.post(
                QWEN_API_URL,
                json=request_data,
                stream=True,
                timeout=60
            )
            response.raise_for_status()
      
            # 用于累积完整的回复
            full_response = ""             
            # 处理流式响应
            for line in response.iter_lines():
                if not line:
                    continue
                    
                try:
                    # 解码二进制数据
                    line_str = line.decode('utf-8')
                    if line_str.startswith('b\''):
                        # 去除b''
                        line_str = line_str[2:-1]
                        # 处理转义字符
                        line_str = bytes(line_str, 'utf-8').decode('unicode_escape')
                    
                    # 解析JSON
                    data = json.loads(line_str)
                    print(f"收到数据行: {data}")  # 调试日志
                    
                    if data.get('done', False):
                        break
                        
                    if 'message' in data:
                        content = data['message'].get('content', '')
                        if content:
                            full_response += content
                            
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {str(e)}, 原始数据: {line}")
                    continue
                except Exception as e:
                    print(f"处理响应行时出错: {str(e)}, 原始数据: {line}")
                    continue

            if not full_response:
                raise ValueError("未收到有效的总结结果")
                
            return full_response
                
        except Exception as e:
            print(f"总结文本失败: {str(e)}")
            raise

    def process_audio_file(self, audio_path: str) -> dict:
        """处理上传的音频文件"""
        try:
            # 转写音频
            transcription = self.transcribe_audio(audio_path)
            if not transcription:
                return {"error": "音频转写结果为空"}

            # 总结文本
            summary = self.summarize_text(transcription)
            
            return {
                "transcription": transcription,
                "summary": summary
            }
        except Exception as e:
            print(f"处理音频文件失败: {str(e)}")
            raise

    def get_ai_response(self, text: str) -> str:
        """获取 AI 回复"""
        try:
            response = self.session.post(
                QWEN_API_URL,
                json={
                    "model": "qwen2.5",
                    "messages": [
                        {"role": "user", "content": text}
                    ],
                    "stream": False
                },
                timeout=(5, 60)
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"获取AI回复失败: {str(e)}")
            raise

    def process_audio_chunk(self, audio_data: bytes) -> str:
        """处理音频数据块"""
        try:
            # 保存临时音频文件
            temp_path = os.path.join(AUDIO_UPLOAD_FOLDER, 'temp_chunk.wav')
            with open(temp_path, 'wb') as f:
                f.write(audio_data)

            # 转写音频
            text = self.transcribe_audio(temp_path)

            # 删除临时文件
            os.remove(temp_path)

            return text
        except Exception as e:
            print(f"处理音频数据块失败: {str(e)}")
            raise
