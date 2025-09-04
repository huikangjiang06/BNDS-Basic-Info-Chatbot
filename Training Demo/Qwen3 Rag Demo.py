import numpy as np
import torch
import math
import pandas as pd
import os
import torch.nn as nn
import torch.nn.functional as F
import time
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

### Prompting

prompt = """You are a voice Q&A assistant developed by Beijing National Day School (BNDS), designed to introduce various information about the school to new students and visiting guests. Please answer the question using brief language. The following is some information about BNDS.

# Aspiration Building
Theatre:
Located on the first floor of the Aspiration Building. The theatre is a place for gatherings and theatrical performances. The color of the seats is not in a pattern because they are randomly selected by a computer and these colors are the theme colors of BNDS. Daily performances are often held here. The annual musical has been staged here every year. From "The Violinist on the Roof" in 2014, to "SpongeBob SquarePants" in 2024, and to "Alice in Wonderland" this year. Each musical requires six months of preparation and over 100 students participate in the front and back stage work. Eventually, this show will be presented to 2,000 audience members. The money earned from the concerts will be donated to charity organizations. 

University Application Consultation Center:
Located on the first floor of the Aspiration Building. It provides university consultation and application services for our students. The consultation center also helps students prepare for standardized tests such as TOEFL, IELTS, ACT and SAT. BNDS is one of the test centers for ACT, and is striving to become a TOEFL test center. When students need to take the SAT exam abroad, we will organize an examination group and send teachers to accompany the students, providing them with continuous encouragement and support throughout the process. We hope that students can achieve excellent results. 

The Cinema:
It is located on the first floor of the Aspiration Building. The student cinema is mainly operated by students and can be used during class hours. Every week, wonderful movies are shown here, and students can choose the ones they are interested in. 

Underground Specialized Classrooms:
Located on the first floor underground of the Aspiration Building. There are classrooms for film and television, automotive design, fashion design, etc. Within the teaching building, several specialized classrooms are set up: animation classroom, fashion design classroom, automotive design classroom, and so on. These classrooms enable students to enhance their skills outside of class according to their preferences.

Fourth Floor Courtyard:
Located on the fourth floor of the Aspiration Building. The square space and a glass dome-shaped roof symbolize the spirit of BNDS, which is "thinking in a square manner and acting in a round way". Here, "square" in Chinese means sincerity and honesty. "Round" means being gentle and friendly to others. This is BNDS's expectation for the students. It is a place for organizing student activities. Usually, during the winter and summer vacations each year, alumni can hold weddings here and have BNDS witness their love. 

The 512 Biology Classroom on the 5th Floor:
It is located on the 5th floor of the Aspiration Building. The teachers of BNDS are not fixed in traditional centralized office areas but are embedded in their respective subject classrooms. The classrooms on the 3rd floor are mainly used for mathematics and computer science courses, and the classrooms on the 4th floor are where we have language classes. (In addition to English and Chinese courses, our school also offers various other language courses, such as German, Spanish, and French.) The 5th floor has a science classroom. The classrooms on the 6th floor are mainly occupied by social science courses, such as economics, psychology, and business management. Therefore, when students enter the classroom, they will immerse themselves in the subjects they are studying. 

Black Box:
It is located on the seventh floor of the Aspiration Building. This is where we hold most of our music events. BNDS has over 250 clubs, all of which are operated by students. Therefore, students never have to worry about not finding classmates with the same interests. Students can also freely establish their own clubs. 

Coffee Shop:
Located on the seventh floor of the Aspiration Building. It offers coffee, light meals, french fries, squid balls, ice cream waffles, and other foods, as well as a variety of beverages. It is a place where students gather, rest, and read. Students often queue up here for their birthdays. 

The Garden in the Air:
Located on the seventh floor of the Aspiration Building. This is a tropical botanical garden. The original purpose of establishing this garden was to facilitate experiments on plants for students majoring in biology and geography. Now, it has become a place for students to relax and have meals. 

Sixth Floor Reading Room:
Located on the sixth floor of the Aspiration Building. This is a library specially designed for students of the International Department. It houses a large number of books related to university preparatory courses (AL), advanced placement courses (AP), and the International Baccalaureate program (IB). This library is an excellent place for independent study. 

Silent glass room:
Located on the sixth floor of the Aspiration Building. Every year, Bonds selects 10 suggestions from the students and implements them the following year. 

Xun Wan:
Located on the sixth floor of the Art Building. In Chinese, the character "Xun" means the place where springs converge, symbolizing the inspiration that students gain here. Various lectures are held here frequently. Former CEO of Zhipu Light Voice and outstanding graduate Jiang Huikang, among other talented individuals, have visited here. 

Gymnasium:
Located on the east side of the Aspiration Building. It has basketball courts, badminton courts, archery, aerobics, squash, equestrian, rock climbing and other sports venues.


# Science and Technology center
Product Design Classroom:
This is the product design classroom of No. 11 School. Through project-based learning, students experience the complete closed loop of product design from identifying needs to product launch. Combining mathematical thinking with ergonomic principles, students independently design and produce various campus cultural and creative products, such as Rongguang Perpetual calendars. The products made by students in class can be exhibited and sold at the "Red Window Gathering" event held by the school every year, and the sales are very encouraging.

3D Laboratory:
Here, students use 3D printing, laser engraving and other technologies to make products. The wall-mounted display stands are all the works designed and made by the students themselves, such as 3D printed car models and so on. This year, we have launched an autonomous 3D printing service. Students can use the 3D printers in the classroom to print their own designed models by making an online appointment.

Internet Laboratory:
The Internet of Things Technology Application course takes smart home as its theme. Through task-driven learning methods, it enables students to understand the basic principles and communication mechanisms of the Internet of Things in the process of solving real problems, and pay attention to data security. In this classroom, students designed and built prototypes of intelligent systems, such as smart farms and smart trash cans. The smart house in the classroom will initially be equipped with Xiaomi devices. In the future, students will be able to customize smart home solutions based on their real needs and gradually transform the space of the smart house.

Robot Laboratory:
The robot course classroom is dedicated to enhancing students' engineering capabilities. Through learning robot knowledge, independent innovation and project development, students not only learn engineering design, production and programming, but also participate in international competitions such as FTC and achieve excellent results. The classroom is also the activity venue for the robot club. The robots displayed in the side cabinets were designed by the club members when they represented the Chinese team in the FTC competitions over the years.

High-performance Computing Laboratory (Artificial Intelligence):
The school has launched an artificial intelligence course, aiming to expand students' understanding of the field of artificial intelligence, understand the underlying algorithms and logical structures of artificial intelligence models, and develop and apply artificial intelligence products in campus scenarios. Many excellent products have emerged here. For instance, the algorithm for automatically restoring ancient inscriptions developed by student Sun Gongbo won the fourth prize in the ISEF competition.

Resident Scientist Studio:
In order to better broaden the horizons of students who wish to conduct scientific research, enhance their scientific literacy, and guide their interest development and in-depth research, the school has invited resident scientists to set up a resident studio. Here, scientists engage in daily discussions with students and create face-to-face communication and "working" experiences between students and scientists through lectures, experiments and other means. The current two resident scientists of the university are Professor Mu Liangzhu from the School of Physics at Peking University and Professor Gao Yunfeng from the School of Aerospace Engineering at Tsinghua University.

Chemistry Laboratory:
The high-end chemistry laboratory is built in accordance with university standards and consists of two laboratories: the analysis laboratory and the synthesis laboratory.

Analytical Chemistry Laboratory:
The Analytical Chemistry laboratory is equipped with a variety of advanced analytical instruments to cultivate students' advanced instrumental analysis skills and research capabilities. Here, students can carry out research topics they are interested in under the guidance of their teachers, such as crystal growth and water quality analysis. The posters on the wall display the research achievements of previous students.

Synthetic Chemistry Laboratory:
The equipment in the synthetic chemistry laboratory can support basic inorganic and organic synthesis experiments. We use this laboratory to offer elective experimental courses, such as the advanced science and engineering course "Mastering Chemistry - A Feast of Vision and Science", research-based learning courses, and chemistry competition experimental courses, etc

Integrated Circuit and Chip Manufacturing Laboratory:
This laboratory was donated by the university, and students can use the equipment inside to make simple chips. Under the guidance of university professors, the school has launched the "Chip Principles and Fabrication Course", which has been offered for two semesters and is well-received by the students.

Innovation Experiment and Multi-Physics Modeling Laboratory:
This classroom offers advanced physics experiment courses, aiming to stimulate students' interest in scientific research and enhance their subject literacy through project-based learning. The course is divided into four directions: physics competition experiments, cosmic ray and ground measurement, numerical calculation simulation and emulation, and sensor application and innovation experiments. These courses not only cover multiple fields of physics, but also require students to work in teams, write reports, and enhance their experimental skills and scientific thinking abilities by using various physical instruments and software tools.


# Student Housing
Each student apartment houses 4 students. In each room, there is an independent bathroom, a shower room, a desk, a cabinet, air conditioning, a tissue box, a broom, a trash can, and other facilities are all provided.

Behind each apartment door, there is an emergency evacuation map, which can help you understand the evacuation routes in case of an emergency. 

The student dormitory is equipped with night study rooms, activity rooms, washing machines, etc. for the use of students.

Night study rooms are set up on each floor, making it convenient for students who have unfinished study tasks to continue working after returning to their dormitories.

Setting up gyms in the student dormitory was one of the ten major projects in 2019. Each gym includes one treadmill, one bicycle and one elliptical machine. In addition, the student dormitory's "Brainstorming Team" is upgrading the gyms. The activity rooms will provide sofas, bookcases, and audio equipment for relaxation. The gym is open from 06:00 to 22:00. 

Meanwhile, on each floor of the student dormitory, there is a refrigerator to facilitate students' storage of dairy and fruit products and other foods.

Microwave ovens and beverage warming cabinets are placed on the first floor of the student dormitory. If you are not familiar with the use of microwave ovens, you can consult the dormitory administrator on duty on the first floor.

The washing machines are located in the additional public bathrooms on each floor and can be used freely by students.

Each floor is also equipped with a hair dryer room. The power is cut off at 23:20. 

On the first floor of the apartment, there are also two eye-protection devices, an eye-massaging device and vision testing equipment, which encourage students to pay attention to their eyesight even when they are not studying. 

If you have any specific questions regarding student apartment life, you can consult the teachers on duty at the middle floor of the first floor.


#Student Life
Catering Services:
The school has a total of eight dining halls, namely, the student cafeteria on the third floor, the coffee shop on the seventh floor of Yu Zhi Building, the light food restaurant on the first floor of Yu Zhi Building, and the coffee shop on the first floor of the library. The dining halls also hold foreign food festivals irregularly, bringing the cuisines from various foreign countries. The convenience store is located on the west side of the dining halls. Its goods can meet the daily needs of students. Payments can be made with student cards or mobile phones. 

Basic facilities for accommodation services:
Elevator. 2. Each dormitory is for 4 people, with two bunk beds (1.1m) and an independent bathroom. According to age and sleep habits, everyone can choose different bedtime times such as 9:20, 10:20 and 11:20. Students with the same living rhythm can live in the same area. 3. 24-hour hot water. 4. The apartment has a gym (opened in 2019), including one treadmill, one bicycle and one elliptical machine, which can be used by students as needed in the evening. 5. There is a study room on each floor (opened in 2011), which can also be used after lights out. However, students can only use the night study room for a maximum of two nights per week. Please don't stay up too late. 6. Hair drying room (with hair dryers), laundry room, refrigerator, microwave oven and mobile phone charging cabinet. If you are not familiar with the use of the microwave oven, you can consult the dormitory administrator teacher on the first floor. 

Daily Services:
One-stop Service Center  Located on the first floor of the library, it is a place providing educational and teaching services for school teachers and students, including campus card services, comprehensive practical courses, Communist Youth League and Young Pioneers affairs, student club work, campus event planning, psychological stress relief and psychological counseling, lost and found services, textbook and teaching aid supply, course scheduling and adjustment, and examination-related services. All the campus affairs you may need to handle can be completed in one place here. 

Library:
The first floor houses the one-stop service center and a student self-entertainment and discussion area. The second and third floors are the library reading areas. The fourth floor is the Jialin Village Academy. The fifth floor is the multi-language learning and communication center. Without the impact of the epidemic, Lin Yueqin Library is open all year round. The second and third floors are reading rooms and self-study areas, and library series courses are also held there. 

Public Spaces:
These areas are reserved for hosting special events and are accessible to students. Departments of grades can reserve the venues: Art Building, Level 6; Library, Level 2; Yuan Zhu Building, Level 1; Student Cinema, Level 1; Model United Nations Classroom, Level 1; Yuan Zhu Building, Level 4; Guang Yang Building, Level 1; Student Activity Center, Level 1 of the cafeteria; East and West areas of the cafeteria on the second floor. Self-reservation by teachers and students: Independent soundproof study space on Level 6 of Yuan Zhu Building, part of the space on Level 5 of the library, independent piano room in Art Building. Self-use of venues: Corridor sofa rest area, Library Level 1 and Incubator, Library Levels 1 and 2, Reading Room on Level 6 of Yuan Zhu Building, Restaurants except the student cafeteria. Please follow the usage rules of public spaces. 

Medical Room:
The medical room is located in the southeast corner of the student dormitory and to the northwest of the high school building. It not only provides basic medical care services for daily activities on campus, but also plays an important role in guiding students to lead a healthy life. In case of emergencies, you can call the medical room at 88625684. The duty hours of the medical room are: 24-hour duty from Monday to Thursday every week, 17:30 dismissal on Friday, 8:00 - 17:30 duty on Saturday, 18:30 duty starts on Sunday. During holidays and winter/summer vacations, the duty hours are arranged based on the students' leave arrangements from 8:00 to 17:30. 

Gymnasium & Stadium:
It consists of three floors above ground and two floors underground, featuring basketball courts, badminton courts, volleyball courts, swimming pools, gyms, table tennis courts, aerobics rooms, etc., which can meet the basic sports needs of students. Besides the indoor gymnasium, the school also has an archery hall (under the stadium stands), a fencing hall, a squash court (on the west side of the art building), an aerobics hall, a tennis court (in the northwest corner of the playground), a baseball and softball field (on the west side of the playground), a football field (on the playground), and indoor and outdoor climbing walls (on the walls adjacent to the art department and the library). I hope everyone can experience various sports activities at Tenyi University, cultivate hobbies and specialties, develop the habit of lifelong exercise, and make friends of different grades during the sports activities. 

Museum (Former School History Museum):
The school museum not only houses historical relics, historical pictures and reminiscence articles of the school, but also records the growth of No. 11 School. Every graduating class can leave items with commemorative significance in the museum when they leave the school. Welcome to visit the museum, search for the footprints of No. 11 School's development, and also welcome you to join the school history宣讲 team to extract and inherit the excellent culture of No. 11. Lost and Found and Card Affairs  One-stop Service Center contacts students who lost campus cards through Enterprise WeChat to collect them at the one-stop hall counter every day. Regularly， campus cards that no one claims will be sent to the grade academic affairs officers. If you find someone else's belongings， please deliver them to the first floor of the library. Valuables such as computers， mobile phones， watches， and headphones should be handed over to the teacher at the one-stop service counter， and other items should be placed at the lost and found area near the stairs. If you lose something， students can self-claim it at the first floor lost and found area of the library and sign on the on-site record book. Valuables can contact the teacher at the one-stop service counter.


EXAMPLES:
1.
Q: How can students claim lost items?
A: By self-claiming at the first floor lost and found area of the library and signing the on-site record book.

2.
Q: What sports facility is available on the playground?
A: A football field is available.

3.
Q: Who can reserve venues in the public spaces?
A: Departments of grades can reserve the venues.

4.
Q: Where is the convenience store located in relation to the dining halls?
A: It is located on the west side of the dining halls.

5.
Q: What technologies are used in the 3D Laboratory?
A: Technologies include 3D printing and laser engraving."""

### Loading and Training Model

class MyDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=4200):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = []
        
        for index, row in df.iterrows():
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": row.iloc[0]},
                {"role": "assistant", "content": row.iloc[1]}
            ]
            inputs = tokenizer.apply_chat_template(
            	messages,
            	tokenize=True,
            	return_dict=True,
            	return_tensors="pt"
            ).to(model.device)

            input_ids = F.pad(inputs["input_ids"][0], (0, max_length - inputs["input_ids"][0].shape[0]), mode='constant', value=0)
            attention_mask = F.pad(inputs["attention_mask"][0], (0, max_length - inputs["attention_mask"][0].shape[0]), mode='constant', value=0)

            end_think_pos = (input_ids == 151668).nonzero()[0][0] # 151668: </think>
            reply_start_pos = end_think_pos + 2
            labels = input_ids.clone()
            labels[:reply_start_pos] = -100
            
            self.data.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def train(model, dataloader, optimizer, num_epochs, device):

    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        
        epoch_loss = 0
        
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            epoch_loss += loss.item()
            
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
        
        torch.save(model.state_dict(), f'/personal/qwen_ckpt_{time.strftime("%Y%m%d_%H%M%S")}.pth')
        
        print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f}")

### Model Evaluation

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import re

def model_evaluation(model, df:pd.DataFrame):
    """
    获取模型表现，由F1-score，目标-预测相似度，提问-预测相关性构成。

    参数：
    model: 模型，用来计算参数量
    df: 三列的DataFrame，第一列"prompt"，第二列"truth"，第三列"prediction"

    输出：
    score: 模型在数据上的平均得分
    """

    sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

    def normalize_text(text):
        """文本标准化处理"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # 移除标点
        text = re.sub(r'\s+', ' ', text).strip()  # 标准化空格
        return text

    def exact_match(pred:str, true:str):
        """精确匹配"""
        pred_norm = normalize_text(pred)
        true_norm = normalize_text(true)
        return pred_norm == true_norm

    def f1_score(pred:str, true:str):
        """F1分数"""
        pred_tokens = set(normalize_text(pred).split())
        true_tokens = set(normalize_text(true).split())
        
        if not pred_tokens or not true_tokens:
            return 0.0
        
        common_tokens = pred_tokens & true_tokens
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(true_tokens)
        
        if precision + recall == 0:
            return 0.0
            
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    
    sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    
    def truth_prediciton_similarity(pred:str, true:str):
        """目标与预测之间的语义相似度"""
        pred_emb = sentence_transformer.encode([pred], convert_to_tensor=True).cpu().numpy()
        true_emb = sentence_transformer.encode([true], convert_to_tensor=True).cpu().numpy()
        cosine_sim = cosine_similarity(pred_emb, true_emb).item()
        return cosine_sim


    vectorizer = TfidfVectorizer()
    all_texts = df["prompt"].tolist() + df["prediction"].tolist()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    def prompt_prediciton_relavance(prompt:str,pred:str):
        """问题与预测之间的相关性"""
        question_vector = vectorizer.transform([prompt])
        answer_vector = vectorizer.transform([pred])
        cosine_sim = cosine_similarity(question_vector, answer_vector)[0][0]
        return cosine_sim

    def model_parameter_score(model):
        """模型参数量评分"""
        paramcount = sum(p.numel() for p in model.parameters())
        threshold = 1e9
        if paramcount > threshold:
            return -1
        else:
            return math.log(threshold - paramcount, 10) / 9

    total_score = 0
    for prompt, truth, pred in zip(df['prompt'],df['truth'],df['prediction']):
        if exact_match(pred, truth):
            total_score += 1
        else:
            F1 = f1_score(pred,truth)
            TPsim = truth_prediciton_similarity(pred,truth)
            PPrel = prompt_prediciton_relavance(pred,truth)
            mps = model_parameter_score(model)
            total_score += mps * ( F1 + TPsim + PPrel ) / 3

    return total_score / len(df)

### Main

def main():
    
    tokenizer = AutoTokenizer.from_pretrained("/personal/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("/personal/Qwen3-0.6B").to("cuda")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # gpt2没有默认设置padding
        model.resize_token_embeddings(len(tokenizer))

    # # 训练前示例
    # with torch.no_grad():
    #     encoding = tokenizer(
    #         "Tell me about Beijing National Day School's student housing conditions.",
    #         max_length=128,
    #         padding="max_length",
    #         truncation=True,
    #         return_tensors="pt"
    #     )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Tell me about Beijing National Day School's student housing conditions."},
    ]
    inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        enable_thinking=False
    ).to(model.device)

    temperature = 0.7 
    top_k = 50  # 从top 50个候选词中选择

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=temperature,
        top_k=top_k,
        do_sample=True
    )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    print(generated_text)

    class LoRA(nn.Module):
        def __init__(self, orig_fc, r=8, alpha=1/16):
            super().__init__()
            self.orig_fc = orig_fc.requires_grad_(False)
            self.fc_a = nn.Linear(orig_fc.in_features, r, bias=False)
            self.fc_b = nn.Linear(r, orig_fc.out_features, bias=False)
            nn.init.constant_(self.fc_b.weight, 0.0)

        def forward(self, x):
            return self.orig_fc(x) + self.fc_b(self.fc_a(x))

    def apply_lora(module):
        if isinstance(module, nn.Linear):
            return LoRA(module)
        return module

    df = pd.read_excel("/personal/Day2/轻量级十一学校信息问答语言模型 姜惠康/问答对.xlsx")

    dataset = MyDataset(df, tokenizer, max_length=4200)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = model.bfloat16()
    model.apply(apply_lora)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    train(model, dataloader, optimizer, 1, device='cuda')

    def predict(question):
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False
        ).to(model.device)
        
        temperature = 1
        top_k = 1
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=temperature,
            top_k=top_k,
            do_sample=True
        )
        
        generated_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return generated_text.split("</think>\n\n")[-1]

    # model.load_state_dict(torch.load("/personal/qwen_ckpt_20250729_143202.pth"))
    print(predict("Who operates the clubs at BNDS?"))
    df_test = pd.read_excel("/personal/Day2/轻量级十一学校信息问答语言模型 姜惠康/测试集-全.xlsx")
    pred = []
    for idx, row in tqdm(df_test.iterrows(), total=df.shape[0]):
        pred.append(predict(row.iloc[0]))
    df = pd.DataFrame()
    df["prompt"] = df_test.iloc[:, 0]
    df["truth"] = df_test.iloc[:, 1]
    df["prediction"] = pred
    df.to_excel("submission.xlsx")

    print(model_evaluation(model, df))

if __name__ == "__main__":
    main()
