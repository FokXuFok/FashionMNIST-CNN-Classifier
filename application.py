import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# 中文类别标签
classes = ('T恤/上衣', '裤子', '套头衫', '连衣裙', '外套',
           '凉鞋', '衬衫', '运动鞋', '包', '短靴')


# 加载模型
def load_model():
    class CNN(torch.nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
            self.bn1 = torch.nn.BatchNorm2d(16)
            self.pool1 = torch.nn.MaxPool2d(2, 2)
            self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
            self.bn2 = torch.nn.BatchNorm2d(32)
            self.pool2 = torch.nn.MaxPool2d(2, 2)
            self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
            self.bn3 = torch.nn.BatchNorm2d(64)
            self.pool3 = torch.nn.MaxPool2d(2, 2)
            self.conv4 = torch.nn.Conv2d(64, 128, 3, padding=1)
            self.bn4 = torch.nn.BatchNorm2d(128)
            self.pool4 = torch.nn.MaxPool2d(2, 2)
            self.conv5 = torch.nn.Conv2d(128, 256, 3, padding=1)
            self.bn5 = torch.nn.BatchNorm2d(256)
            self.pool5 = torch.nn.MaxPool2d(1, 1)
            self.fc1 = torch.nn.Linear(256, 128)
            self.fc2 = torch.nn.Linear(128, 64)
            self.fc3 = torch.nn.Linear(64, 10)

        def forward(self, x):
            x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
            x = self.pool1(x)
            x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
            x = self.pool2(x)
            x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
            x = self.pool3(x)
            x = torch.nn.functional.relu(self.bn4(self.conv4(x)))
            x = self.pool4(x)
            x = torch.nn.functional.relu(self.bn5(self.conv5(x)))
            x = self.pool5(x)
            x = x.view(x.size(0), -1)
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model_files = [
        '5layer_cnn_labelsmooth_best_model.pth',
        '5layer_cnn_labelsmooth_final_model.pth',
        'best_model.pth',
        'model.pth'
    ]

    model_path = None
    for file in model_files:
        if os.path.exists(file):
            model_path = file
            break

    if not model_path:
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN()

    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        return None, None


# 预测图片，返回结果字典
def predict_image(image_path, model, device):
    try:
        image = Image.open(image_path)

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_class = torch.max(probabilities, 1)

        predicted_class = classes[top_class.item()]
        confidence = top_prob.item() * 100
        probs = probabilities.squeeze().cpu().numpy()

        result = {
            'predicted': predicted_class,
            'confidence': confidence,
            'probs': {classes[i]: float(probs[i] * 100) for i in range(len(classes))}
        }
        return result

    except Exception as e:
        return None


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("FashionMNIST 服装识别系统")
        self.root.geometry("700x600")
        self.root.resizable(False, False)

        self.model = None
        self.device = None
        self.current_image_path = None

        self._build_ui()
        self._load_model()

    def _build_ui(self):
        # 标题
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        tk.Label(title_frame, text="FashionMNIST 服装识别系统",
                 font=("Microsoft YaHei", 18, "bold"), fg="white", bg="#2c3e50").pack(expand=True)

        # 主体区域
        body = tk.Frame(self.root, padx=20, pady=10)
        body.pack(fill=tk.BOTH, expand=True)

        # 左侧：图片显示
        left_frame = tk.LabelFrame(body, text="图片预览", font=("Microsoft YaHei", 10), padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))

        self.image_label = tk.Label(left_frame, width=30, height=18, bg="#ecf0f1",
                                    relief=tk.SUNKEN, text="请选择图片", font=("Microsoft YaHei", 9))
        self.image_label.pack()

        # 右侧：结果区域
        right_frame = tk.Frame(body)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 预测结果
        result_frame = tk.LabelFrame(right_frame, text="识别结果", font=("Microsoft YaHei", 10), padx=10, pady=10)
        result_frame.pack(fill=tk.X, pady=(0, 10))

        self.result_label = tk.Label(result_frame, text="等待识别...",
                                     font=("Microsoft YaHei", 14, "bold"), fg="#7f8c8d")
        self.result_label.pack(pady=5)

        self.confidence_label = tk.Label(result_frame, text="",
                                         font=("Microsoft YaHei", 10), fg="#95a5a6")
        self.confidence_label.pack()

        # 概率分布
        prob_frame = tk.LabelFrame(right_frame, text="各类别概率", font=("Microsoft YaHei", 10), padx=10, pady=5)
        prob_frame.pack(fill=tk.BOTH, expand=True)

        self.prob_bars = {}
        for cls in classes:
            row = tk.Frame(prob_frame)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=cls, font=("Microsoft YaHei", 9), width=8, anchor=tk.W).pack(side=tk.LEFT)
            bar = ttk.Progressbar(row, length=150, mode='determinate')
            bar.pack(side=tk.LEFT, padx=5)
            pct_label = tk.Label(row, text="0.0%", font=("Microsoft YaHei", 8), width=7, anchor=tk.E)
            pct_label.pack(side=tk.LEFT)
            self.prob_bars[cls] = (bar, pct_label)

        # 底部按钮
        btn_frame = tk.Frame(self.root, padx=20, pady=10)
        btn_frame.pack(fill=tk.X)

        self.btn_select = tk.Button(btn_frame, text="选择图片", font=("Microsoft YaHei", 10),
                                    bg="#3498db", fg="white", width=12, command=self._select_image)
        self.btn_select.pack(side=tk.LEFT, padx=5)

        self.btn_predict = tk.Button(btn_frame, text="开始识别", font=("Microsoft YaHei", 10),
                                     bg="#27ae60", fg="white", width=12, command=self._predict,
                                     state=tk.DISABLED)
        self.btn_predict.pack(side=tk.LEFT, padx=5)

        self.btn_batch = tk.Button(btn_frame, text="批量识别", font=("Microsoft YaHei", 10),
                                   bg="#e67e22", fg="white", width=12, command=self._batch_predict)
        self.btn_batch.pack(side=tk.LEFT, padx=5)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = tk.Label(self.root, textvariable=self.status_var, font=("Microsoft YaHei", 8),
                              fg="#95a5a6", anchor=tk.W, padx=10)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _load_model(self):
        self.status_var.set("正在加载模型...")
        self.root.update()
        self.model, self.device = load_model()
        if self.model is not None:
            self.status_var.set(f"模型加载成功 | 设备: {self.device}")
        else:
            self.status_var.set("模型加载失败，请检查模型文件是否存在")
            messagebox.showerror("错误", "找不到模型文件，请确保 .pth 文件在当前目录下")

    def _select_image(self):
        filetypes = [("图片文件", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp")]
        filepath = filedialog.askopenfilename(title="选择图片", filetypes=filetypes)
        if filepath:
            self.current_image_path = filepath
            self._display_image(filepath)
            self.btn_predict.config(state=tk.NORMAL)
            self.result_label.config(text="等待识别...", fg="#7f8c8d")
            self.confidence_label.config(text="")
            self.status_var.set(f"已选择: {os.path.basename(filepath)}")

    def _display_image(self, filepath):
        try:
            img = Image.open(filepath)
            # 缩放显示
            img.thumbnail((250, 300), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # 保持引用
        except Exception as e:
            messagebox.showerror("错误", f"无法打开图片: {e}")

    def _predict(self):
        if not self.current_image_path or self.model is None:
            return

        self.status_var.set("正在识别...")
        self.root.update()

        result = predict_image(self.current_image_path, self.model, self.device)
        if result:
            self._show_result(result)
            self.status_var.set(f"识别完成: {result['predicted']} ({result['confidence']:.1f}%)")
        else:
            messagebox.showerror("错误", "识别失败")
            self.status_var.set("识别失败")

    def _show_result(self, result):
        self.result_label.config(text=f"{result['predicted']}", fg="#2c3e50")
        self.confidence_label.config(text=f"置信度: {result['confidence']:.1f}%")

        max_prob = max(result['probs'].values()) if result['probs'].values() else 1
        for cls in classes:
            bar, pct_label = self.prob_bars[cls]
            prob = result['probs'].get(cls, 0)
            bar['value'] = prob
            pct_label.config(text=f"{prob:.1f}%")
            if cls == result['predicted']:
                pct_label.config(fg="#e74c3c", font=("Microsoft YaHei", 8, "bold"))
            else:
                pct_label.config(fg="#333333", font=("Microsoft YaHei", 8))

    def _batch_predict(self):
        pictures_folder = './图片'
        if not os.path.exists(pictures_folder):
            messagebox.showwarning("提示", "找不到'图片'文件夹，请确保当前目录下存在该文件夹")
            return

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
        image_files = [f for f in os.listdir(pictures_folder)
                       if any(f.lower().endswith(ext) for ext in image_extensions)]

        if not image_files:
            messagebox.showinfo("提示", "'图片'文件夹中没有图片文件")
            return

        if self.model is None:
            messagebox.showerror("错误", "模型未加载")
            return

        # 创建批量结果窗口
        batch_win = tk.Toplevel(self.root)
        batch_win.title("批量识别结果")
        batch_win.geometry("500x400")

        text = tk.Text(batch_win, font=("Microsoft YaHei", 10), padx=10, pady=10)
        scroll = tk.Scrollbar(batch_win, command=text.yview)
        text.config(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(fill=tk.BOTH, expand=True)

        text.insert(tk.END, f"共找到 {len(image_files)} 张图片，开始识别:\n\n")

        for img_file in image_files:
            filepath = os.path.join(pictures_folder, img_file)
            result = predict_image(filepath, self.model, self.device)
            if result:
                line = f"{img_file}  ->  {result['predicted']}  ({result['confidence']:.1f}%)\n"
                text.insert(tk.END, line)
            else:
                text.insert(tk.END, f"{img_file}  ->  识别失败\n")

        text.insert(tk.END, f"\n识别完成！")
        text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
