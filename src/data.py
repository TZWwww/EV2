import os
import pdb
import random

import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import json
from transformers import AutoTokenizer
import string


def process_question_and_choices(question, choices):
    # Split the question into words
    words = question.split()

    # Extract the second and the last word (e0 and e1)
    if len(words) >= 2:
        e0 = words[1]
        e1 = words[-1][:-1]
    else:
        return "Invalid question format", []

    # Define the new question and choices templates based on the input choices
    if choices == ["Causes", "IsResult", "Vague"]:
        new_question = f"Which is the causal relationship between {e0} and {e1}?"
        new_choices = [f"{e0} causes {e1}.", f"{e0} is result of {e1}.", f"There is no obvious causal relationship between {e0} and {e1}."]
    elif choices == ["Before", "After", "Vague"]:
        new_question = f"Which is the temporal relationship between {e0} and {e1}?"
        new_choices = [f"{e0} is before {e1}.", f"{e0} is after {e1}.", f"There is no obvious temporal relationship between {e0} and {e1}."]
    elif choices == ["HasSubevent", "IsSubevent", "Vague"]:
        new_question = f"Which is the subordinate relationship between {e0} and {e1}?"
        new_choices = [f"{e1} is subevent of {e0}.", f"{e0} is subevent of {e1}.", f"There is no obvious subordinate relationship between {e0} and {e1}."]
    else:
        return "Invalid choices format", []

    return new_question, new_choices


def find_and_sample(A, B):
    # 替换A中所有子列表中的字符串中的空格为下划线
    A = [[s.replace(' ', '_') for s in sublist] for sublist in A]

    # 找到B中每个字符串在A的子列表中的索引
    indexes = set()
    for b in B:
        for i, sublist in enumerate(A):
            if b in sublist:
                indexes.add(i)
                break

    # 对于A中的每个子列表，如果其索引不在找到的索引列表中，则随机选择一个字符串
    C = []
    for i, sublist in enumerate(A):
        if i not in indexes:
            for j in sublist:
                C.append(j)
    C=random.sample(C, 20-len(B))
    # 将B中的内容随机插入到C中
    for b in B:
        insert_index = random.randint(0, len(C))
        C.insert(insert_index, b)

    return C



class EV2Dataset(Dataset):
    def __init__(self,
            task_name,
            data_path,
            max_seq_length=512,
            k=None,
            i=None,
            shuffle=False,
            split=False,
            cluster_dir=None,
            ):
        
        self.i = i
        self.k = k

        with open(data_path, 'r') as f:
            dataset = [json.loads(d) for d in f.readlines()]

        self.task_name = task_name

        self.formats = {
                "S_CEC" : self.S_CEC,
                "I_CEC" : self.I_CEC,
                "I_CEC_DIRECT" : self.I_CEC_DIRECT,
                "CEC_rq3" : self.CEC_rq3,

                "S_CRR": self.S_CRR,
                "I_CRR" : self.I_CRR,
                "I_CRR_DIRECT" : self.I_CRR_DIRECT,
                "CRR_rq3" : self.CRR_rq3,
                }
        
        if cluster_dir is not None:
            REL_MAP = {'Precedence': 'after', 'Succession': 'before', 'Synchronous': 'simultaneously', 'Reason': 'cause', 'Result': 'effect', 'Condition': 'isCondition', 'Contrast': 'isContrast', 'Instantiation': 'isInstantiation', 'Restatement': 'isRestatement', 'Alternative': 'isAternative', 'Exception': 'isException'}
            with open(cluster_dir) as f: self.clusters = json.load(f)

        self.src_inputs_list = []
        self.labs = []
        self.src_list = []
        self.pos_list = []
        self.stm_list = []

        ind = 0
        for n, data in tqdm(enumerate(dataset)):

            data_generator = self.formats[task_name](data)

            for src, stms, lab in data_generator:

                if src is None:
                    continue

                src_list = []
                stm_list = []
                for src_, stm_ in zip(src, stms):
                    if ind < 5: 
                        print('================src================')
                        print(src_)
                        print('================lab================')
                        print(lab)
                        print()

                    src_list.append(src_)
                    stm_list.append(stm_)

                self.src_list.append(src_list)
                self.stm_list.append(stm_list)
                self.labs.append(lab)
                ind += 1


    def build_demonstrations(self, doc_indices):

        seq = ''
        # seq = 'Examples:\n\n' # vicuna
        for ind in doc_indices:
            state = None
            ind_ = ind
            while state is None:
                state, _ = self.wrap_data(self.trainset[ind_], add_lab=True)
                ind_ += 1

            seq += state
            seq += '\n\n' 

        return seq


    def S_CEC(self, data):

        prompt = 'Instructions:\nAnswer the question by selecting A, B, C, D.\n\n'
        # prompt = 'Answer the question using only the letter label of the option.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem
        lab2ind = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        choices = data['choices']
        context = data['context']
        question = data['question']
        lab=lab2ind[choices.index(data['e1'])]
        inds = [char for char in string.ascii_letters if char.isupper()]
        ipt = ''.join([f'{inds[i]}. {c}\n' for i, c in enumerate(choices)])

        # unified prompt
        if context:
            src = f'Context:\n{context}\nQuestion:\n{question}\nChoices:\n{ipt}\nThe answer is'
        else:
            src = f'Question:\n{question}\nChoices:\n{ipt}\nThe answer is'

        # src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [['A', 'B', 'C', 'D']], lab
    
    def I_CEC(self, data):

        formatted_string = "\n".join([f"{key}:\n{value}" for key, value in data["instances"].items()])
        formatted_string="Instances:\n"+formatted_string+"\n"
        
        context = data['context']
        if context:
            prompt = 'Instructions:\nAnswer the question by selecting A, B, C, D. Note that all events appearing in \"Context\", \"Question\", and \"Choices\" refer to the specific events described in \"Instances\".\n\n'+formatted_string+"\n\n"
        else:
            prompt = 'Instructions:\nAnswer the question by selecting A, B, C, D. Note that all events appearing in \"Question\", and \"Choices\" refer to the specific events described in \"Instances\".\n\n'+formatted_string+"\n\n"
        # prompt = 'Answer the question using only the letter label of the option.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem
        lab2ind = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        choices = data['choices']
        
        question = data['question']
        lab=lab2ind[choices.index(data['e1'])]
        inds = [char for char in string.ascii_letters if char.isupper()]
        ipt = ''.join([f'{inds[i]}. {c}\n' for i, c in enumerate(choices)])

        # unified prompt
        if context:
            src = f'Context:\n{context}\nQuestion:\n{question}\nChoices:\n{ipt}\nThe answer is'
        else:
            src = f'Question:\n{question}\nChoices:\n{ipt}\nThe answer is'

        # src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [['A', 'B', 'C', 'D']], lab
    

    def I_CEC_DIRECT(self, data):

        ets = data['event_type']

        formatted_string = "\n".join([f"{key}:\n{value}" for key, value in data["instances"].items()])
        formatted_string="Instances:\n"+formatted_string+"\n"
        
        context = data['context']
        if context:
            prompt = 'Instructions:\nAnswer the question by selecting A, B, C, D. Note that all events appearing in \"Context\", \"Question\", and \"Choices\" refer to the specific events described in \"Instances\".\n\n'+formatted_string+"\n\n"
        else:
            prompt = 'Instructions:\nAnswer the question by selecting A, B, C, D. Note that all events appearing in \"Question\", and \"Choices\" refer to the specific events described in \"Instances\".\n\n'+formatted_string+"\n\n"
        # prompt = 'Answer the question using only the letter label of the option.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem
        lab2ind = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        choices = data['choices']
        
        question = data['question']
        lab=lab2ind[choices.index(data['e1'])]
        inds = [char for char in string.ascii_letters if char.isupper()]
        ipt = ''.join([f'{inds[i]}. {c}\n' for i, c in enumerate(choices)])

        # unified prompt
        if context:
            src = f'Context:\n{context}\nQuestion:\n{question}\nChoices:\n{ipt}\nThe answer is'
        else:
            src = f'Question:\n{question}\nChoices:\n{ipt}\nThe answer is'

        # src, lab = self.wrap_data(data)
        prompt += src

        for eid, et in ets.items():
            prompt = prompt.replace(eid, et)

        srcs = [prompt]

        yield srcs, [['A', 'B', 'C', 'D']], lab
    
    def CEC_rq3(self, data):

        formatted_string = "\n".join([f"{key}:\n{value}" for key, value in data["instances"].items()])
        formatted_string="Instances:\n"+formatted_string
        
        prompt = "Instructions:\nIn a scenario explained by the \"Context\", the \"Question\" ask about selecting one event that has a certain event relationship with a particular event, from all possible tail events provided by the \"Choices\", and the \"Instances\" explain all the events in detail with a few sentences. Event semantic knowledge refers to the abstract event types to which specific events belong, and the relationships between these abstract event types. Please output the event semantic knowledge used in solving the following problem. Note that all possible abstract event types in the \"Schema\", and the relationships between abstract events include HasSubevent, IsSubevent, Before, After, Causes, and IsResult. For the tuple [event0, relation, event1], HasSubevent indicates that event1 is a subevent of event0, IsSubevent indicates that event0 is a subevent of event1, Before indicates that event0 occurs before event1, After indicates that event0 occurs after event1, Causes indicates that event0 causes event1, and IsResult indicates that event0 is the result of event1. Output in JSON format, don't generate other sentences."+"\n\n"
        # prompt = 'Answer the question using only the letter label of the option.\n\n'
        prompt += "Requirements:\nAbstract event types can only be chosen from \"Schema\", and the relationships of abstract event types can only be selected from HasSubevent, IsSubevent, Before, After, Causes, and IsResult. Follow the format in examples, output in JSONL format. The key \"event type\" should correspond to a value that is a dictionary with events as keys and their abstracted categories as values. The key \"event relation\" should correspond to a value that is a list of tuples [event0, relation, event1]. The relationships between events include HasSubevent, IsSubevent, Before, After, Causes, and IsResult.\n\n"
        
        # A = [['obstruct justice'], ['make decision'], ['alert'], ['eat breakfast', 'eat breakfast'], ['enhancing technique'], ['celebrate'], ['travel'], ['preparation material'], ['study'], ['conflict'], ['lost job'], ['shape dough', 'knead dough'], ['experience delay'], ['planting flowers'], ['purchase ingredients'], ['submit paper'], ['take break'], ['publication'], ['fly kite'], ['abandon'], ['becoming tired', 'tired', 'feel tired'], ['communication'], ['start moving'], ['shower', 'have shower'], ['get money'], ['slumber', 'sleep'], ['seek help', 'seeking assistance', 'seek assistance'], ['learn', 'learn how'], ['feeling excitement', 'excited'], ['burying cat', 'bury cat'], ['inhaling'], ['examination', 'evaluate'], ['perform'], ['play music', 'hear music'], ['agreeing with'], ['continue dancing'], ['play sport', 'play sports'], ['recover illness'], ['selecting casket', 'choosing casket'], ['exercise', 'exercising'], ['healthcare policy'], ['employment'], ['landscape gardening', 'landscaping garden'], ['conference presentation'], ['tie shoelaces'], ['engage in conversation'], ['check weather'], ['write poem', 'write poetry'], ['pursue education', 'receive education'], ['experience contentment'], ['plan trip'], ['make bread'], ['misunderstanding', 'confusion'], ['repair kite'], ['purchase snacks'], ['analyzing data', 'analyze data'], ['bake cake'], ['hearing news'], ['dry'], ['award ceremony'], ['making friends'], ['eating', 'eat'], ['playing video games'], ['community engagement'], ['passing class'], ['drink more'], ['paint artwork'], ['getting contract'], ['attend class'], ['watch nature'], ['escape consequences'], ['planning strategy'], ['prepare documentation'], ['dream'], ['create'], ['record'], ['climb tree'], ['drink refreshment'], ['academic collaboration'], ['preheat oven'], ['shop for groceries'], ['building cathedral'], ['analyse', 'analysing'], ['plead guilty'], ['digging hole'], ['read book'], ['achievement', 'accomplishment'], ['take a walk'], ['improve skill', 'improve skills'], ['swim'], ['floorspace'], ['purchase supplies'], ['hear singing'], ['observe change'], ['evening concludes'], ['submitting exam'], ['family expansion'], ['financial concerns'], ['feeling lonely'], ['reflect', 'think'], ['organize garden party'], ['follow recipe', 'following recipe'], ['repair vehicle'], ['bricks'], ['complete homework'], ['eat ice cream'], ['gaining weight'], ['lose consciousness'], ['obtain kite', 'get kite'], ['gain knowledge', 'gaining knowledge'], ['distraction'], ['decompose'], ['reacting emotionally'], ['use yeast'], ['protest begins'], ['prepare food', 'prepare meal'], ['kick feet'], ['fall'], ['common interests'], ['visit market'], ['stretching'], ['express information'], ['leave reception'], ['legal action initiation'], ['teach baking classes'], ['go outside'], ['overwhelmed'], ['play continue'], ['commit crime'], ['choose recipe'], ['depart honeymoon'], ['traveling abroad'], ['breathe'], ['daydreaming'], ['rest prepare'], ['contract termination'], ['feeling motivated', 'feel motivated'], ['doing housework'], ['attending commencement'], ['recovering from injury'], ['feel pain'], ['attend afterparty'], ['kill'], ['cooking'], ['receive feedback'], ['hunt'], ['acquiring sponsorship'], ['medication'], ['extinguishing lights'], ['cultural appreciation'], ['choose clothes'], ['legal complication'], ['cleaning'], ['have party'], ['take stand'], ['recover quickly'], ['forming relationships'], ['experience emotion'], ['get weapon'], ['drink energy drink'], ['avoid distractions'], ['revise'], ['becoming hungry'], ['celebrating success', 'celebrate success'], ['receiving financial aid'], ['study more'], ['entertainment break'], ['heal'], ['finishing projects'], ['sweating'], ['life milestone'], ['feel cold'], ['have dinner'], ['teach', 'teaching'], ['formalizing partnership'], ['publish book'], ['overlook'], ['open space'], ['destroy evidenceescape captivity'], ['social interaction'], ['provide assistance'], ['write grant proposal'], ['enjoy breeze'], ['shop'], ['miss deadline'], ['plan future'], ['argue case'], ['legislation'], ['stress relief'], ['physical endurance'], ['memorizing'], ['signing documents'], ['crying'], ['maintenance disruption'], ['withdraw money'], ['graduating'], ['consider consequences'], ['volunteering service'], ['inauguration celebration'], ['achieving academic recognition'], ['doubt', 'ignorance'], ['play'], ['economic impact'], ['feeling energized'], ['external acknowledgement'], ['schedule change'], ['inaugurate ceremony'], ['show rehabilitation'], ['get injuried'], ['brain'], ['receive applause'], ['explore'], ['happy', 'feel happy'], ['watch clouds'], ['dancing'], ['respond to questions'], ['attend seminar'], ['participate event'], ['application process'], ['expansion'], ['teach others'], ['recognize individuals'], ['pay'], ['cool down'], ['acknowledge support'], ['resolve conflict'], ['break law'], ['live'], ['improve health'], ['seek therapy'], ['compose'], ['grieve'], ['career planning'], ['run'], ['being ablebodied'], ['labour'], ['have string'], ['launch investigation'], ['seek entertainment'], ['relaxation'], ['reconciliation'], ['intensifying music'], ['get flour'], ['apathy'], ['destruction'], ['lung'], ['getting job'], ['design recipe'], ['wander'], ['reviewing news'], ['achieving perfection'], ['study recipe'], ['seek approval'], ['negotiating'], ['supporting infrastructure'], ['discuss performance'], ['close attention'], ['interact with audience'], ['pack up kite'], ['submit evidence'], ['return home'], ['physical exertion'], ['engage attention'], ['study sessions'], ['escape captivity'], ['feel discomfort'], ['plan vacation'], ['planning', 'plan'], ['become engrossed'], ['fallAsleep'], ['continuing education'], ['eat cake'], ['introduction'], ['organize materials'], ['buy'], ['move feet'], ['planning confusion'], ['pass out'], ['measure ingredients'], ['initiate project'], ['cdeact'], ['increasing skill'], ['cause disruption'], ['discuss lessons'], ['disappointed with socializing'], ['meal'], ['receive communication'], ['cogitate'], ['loss'], ['emotional support'], ['common ground'], ['read'], ['delegating tasks'], ['overlooking data'], ['sing'], ['realization'], ['escalation of disputes'], ['submit assignment'], ['understanding'], ['die'], ['understanding better'], ['dance celebration'], ['switch roles'], ['being hired'], ['seek employment'], ['prepare breakfast'], ['evaluation knowledge'], ['healthcare demand'], ['personal satisfaction'], ['decorate cake'], ['execute action'], ['isolation'], ['abolish'], ['mix ingredients', 'prepare ingredients', 'get ingredients'], ['studying'], ['exhale'], ['watch kite'], ['tie shoes'], ['accept'], ['summarize'], ['being rejected'], ['anxiety'], ['sharing knowledge'], ['receive award'], ['change of interest'], ['pack supplies'], ['landscaping'], ['financial decision'], ['renew'], ['repare oven'], ['lose balance'], ['being nice'], ['plan travel'], ['distract'], ['improving technique', 'improving techniques'], ['incite conflict'], ['rest after activity'], ['refreshment'], ['talk', 'discuss'], ['eat snack'], ['sadness'], ['listen', 'remember'], ['exhaustion'], ['use breadmaker'], ['implement policy'], ['develop skills'], ['coordinate'], ['indifference'], ['sick'], ['invent'], ['apply'], ['discover'], ['resting', 'rest'], ['trust'], ['observation'], ['implement'], ['help'], ['ponder'], ['source ingredients'], ['leave event'], ['construct'], ['learning'], ['inquiry'], ['feeling invigorated'], ['adjust kite'], ['review'], ['investigate'], ['fall asleep'], ['disinterest'], ['discuss strategy'], ['physical exercise'], ['disappointment'], ['gather information'], ['prepare equipment'], ['cook'], ['disagreement'], ['take exam'], ['prepare presentation'], ['observe weather'], ['breathing'], ['misfortune']]
        A = self.clusters
        B = list(data["event_type"].values())
        combined = find_and_sample(A, B)
        schema = ", ".join(combined)
        prompt += "Schema:\n"+schema+"\n\n"
        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem
        prompt += "Here are some examples:\n\n"+"Instances:\nevent53:\nThis revelation ultimately prompted an individual in the courtroom audience to discretely exit the room and later that evening, the same individual, driven by fear of exposure, went on to commit a fatal assault against a witness who could connect him to the crime.\
\nevent67:\nWhile Mr. Smith was providing his account, he mentioned a key detail that was previously overlooked—a unique tattoo that he glimpsed on the perpetrator's arm.\
\nevent91:\nTwo weeks later, during the heated court proceedings at the downtown courthouse, the homeowner, Mr. Smith, was called to testify before the jury regarding the night of the incident.\
\nevent64:\nIn a quiet suburban neighborhood, a burglary occurred at the Smith residence, where an unknown assailant broke in and stole valuable heirlooms late at night.\
\nevent51:\nUpon hearing the new testimony about the tattoo, a juror who happened to have an interest in body art quietly made a mental note to research the design's origins, intrigued by its possible cultural significance.\
\nevent13:\nUpon hearing Mr. Smith's testimony, one juror with claustrophobia experienced a severe panic attack, which led to the court session being temporarily adjourned as the juror was rushed to the hospital for medical attention.\
\nevent90:\nIn a surprising turn of events during the tea break, the court stenographer, overwhelmed by guilt, approached the judge and admitted to tampering with court transcripts in a previous unrelated case, sparking an investigation into judicial misconduct.\
\n\nContext:\n\"event64\" is before \"event91\". \"event67\" is a subevent of \"event91\".\
\nQuestion:\nWhich is caused by \"event67\"?\
\nChoices:\nA. event13\
\nB. event90\
\nC. event53\
\nD. event51\
\n\nEvent type and event relation:\n{\"event_type\": {\"event67\": \"talk\", \"event53\": \"kill\", \"event64\": \"commit_crime\", \"event91\": \"take_stand\", \"event51\": \"hide_evidence\", \"event13\": \"escape\", \"event90\": \"confess\"}, \"event_relation\": [[\"commit_crime\", \"Before\", \"take_stand\"], [\"take_stand\", \"HasSubevent\", \"talk\"], [\"talk\", \"Causes\", \"kill\"]]}"+"\n\n"
        # prompt += "Examples:\n"+"""{'event_type': {'event67': 'talk', 'event53': 'kill', 'event64': 'commit_crime', 'event91': 'take_stand'}, 'event_relation': [['event64', 'Before', 'event91'], ['event91', 'HasSubevent', 'event67'], ['event67', 'Causes', 'event53']]}"""+"\n\n"
         
        prompt += "Now, based on the above, please output the event semantic knowledge used in solving the following problem.\n\n"                                                       
        prompt += formatted_string+'\n\n'
        
        choices = data['choices']
        context = data['context']
        question = data['question']
        lab = {}
        lab["event_type"]=data["event_type"]
        lab["event_relation"]=data["event_relation"]
        inds = [char for char in string.ascii_letters if char.isupper()]
        ipt = ''.join([f'{inds[i]}. {c}\n' for i, c in enumerate(choices)])

        # unified prompt
        src = f'Context:\n{context}\nQuestion:\n{question}\nChoices:\n{ipt}\nEvent type and event relation:'
        # src, lab = self.wrap_data(data)
        prompt += src

        srcs = [prompt]

        yield srcs, [['A', 'B', 'C', 'D']], json.dumps(lab)

    
    def S_CRR(self, data):

        prompt = 'Instructions:\nAnswer the question by selecting A, B or C.\n\n'
        # vicuna
        # prompt = 'Determine the type of causal relationship between events by returning A, B or C.\n\n'

        # prompt = 'Answer the question using only the letter label of the option.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        # choices = data['choices']
        lab_map={"HasSubevent":"A", "IsSubevent":"B","Before":"A",  "After":"B","Causes":"A", "IsResult":"B","Vague" : "C"}
        choices = data['choices']
        context = data['context']
        question = data['question']
        # pdb.set_trace()
        
        new_question, new_choices=process_question_and_choices(question,choices)
        inds = [char for char in string.ascii_letters if char.isupper()]
        ipt = ''.join([f'{inds[i]}. {c}\n' for i, c in enumerate(new_choices)])
        
        lab=lab_map[data['rel']]
        # src, lab = self.wrap_data(data)
        if context:
            src=f'Context:\n{context}\nQuestion:\n{new_question}\nChoices:\n{ipt}\nThe answer is'
        else:
            src=f'Question:\n{new_question}\nChoices:\n{ipt}\nThe answer is'

        prompt += src

        srcs = [prompt]

        yield srcs, [new_choices], lab
    
    def I_CRR(self, data):
        
        
        formatted_string = "\n".join([f"{key}:\n{value}" for key, value in data["instances"].items()])
        formatted_string="Instances:\n"+formatted_string+"\n"
        
        context = data['context']

        if context:
            prompt = 'Instructions:\nAnswer the question by selecting A, B or C. Note that all events appearing in \"Context\", \"Question\", and \"Choices\" refer to the specific events described in \"Instances\".\n\n'+formatted_string+"\n\n"
        else:
            prompt = 'Instructions:\nAnswer the question by selecting A, B or C. Note that all events appearing in \"Question\", and \"Choices\" refer to the specific events described in \"Instances\".\n\n'+formatted_string+"\n\n"

        # vicuna
        # prompt = 'Determine the type of causal relationship between events by returning A, B or C.\n\n'

        # prompt = 'Answer the question using only the letter label of the option.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        # choices = data['choices']
        lab_map={"HasSubevent":"A", "IsSubevent":"B","Before":"A",  "After":"B","Causes":"A", "IsResult":"B","Vague" : "C"}
        choices = data['choices']
        
        question = data['question']
        new_question, new_choices=process_question_and_choices(question,choices)
        inds = [char for char in string.ascii_letters if char.isupper()]
        ipt = ''.join([f'{inds[i]}. {c}\n' for i, c in enumerate(new_choices)])
        
        lab=lab_map[data['rel']]
        # src, lab = self.wrap_data(data)
        if context:
            src=f'Context:\n{context}\nQuestion:\n{new_question}\nChoices:\n{ipt}\nThe answer is'
        else:
            src=f'Question:\n{new_question}\nChoices:\n{ipt}\nThe answer is'


        prompt += src

        srcs = [prompt]

        yield srcs, [new_choices], lab
    
    def I_CRR_DIRECT(self, data):
        
        ets = data['event_type']
        
        formatted_string = "\n".join([f"{key}:\n{value}" for key, value in data["instances"].items()])
        formatted_string="Instances:\n"+formatted_string+"\n"
        
        context = data['context']

        if context:
            prompt = 'Instructions:\nAnswer the question by selecting A, B or C. Note that all events appearing in \"Context\", \"Question\", and \"Choices\" refer to the specific events described in \"Instances\".\n\n'+formatted_string+"\n\n"
        else:
            prompt = 'Instructions:\nAnswer the question by selecting A, B or C. Note that all events appearing in \"Question\", and \"Choices\" refer to the specific events described in \"Instances\".\n\n'+formatted_string+"\n\n"

        # vicuna
        # prompt = 'Determine the type of causal relationship between events by returning A, B or C.\n\n'

        # prompt = 'Answer the question using only the letter label of the option.\n\n'

        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem

        # choices = data['choices']
        lab_map={"HasSubevent":"A", "IsSubevent":"B","Before":"A",  "After":"B","Causes":"A", "IsResult":"B","Vague" : "C"}
        choices = data['choices']
        
        question = data['question']
        new_question, new_choices=process_question_and_choices(question,choices)
        inds = [char for char in string.ascii_letters if char.isupper()]
        ipt = ''.join([f'{inds[i]}. {c}\n' for i, c in enumerate(new_choices)])
        
        lab=lab_map[data['rel']]
        # src, lab = self.wrap_data(data)
        if context:
            src=f'Context:\n{context}\nQuestion:\n{new_question}\nChoices:\n{ipt}\nThe answer is'
        else:
            src=f'Question:\n{new_question}\nChoices:\n{ipt}\nThe answer is'


        prompt += src

        for eid, et in ets.items():
            prompt = prompt.replace(eid, et)

        srcs = [prompt]

        yield srcs, [new_choices], lab

    
    def CRR_rq3(self, data):
        formatted_string = "\n".join([f"{key}:\n{value}" for key, value in data["instances"].items()])
        formatted_string="Instances:\n"+formatted_string
        
        prompt = "Instructions:\nIn a scenario explained by the \"Context\", the \"Question\" ask about selecting one relationship between two events, from all possible relationships provided by the \"Choices\", and the \"Instances\" explain all the events in detail with a few sentences. Event semantic knowledge refers to the abstract event types to which specific events belong, and the relationships between these abstract event types. Please output the event semantic knowledge used in solving the following problem. Note that all possible abstract event categories in the \"Schema\", and the relationships between abstract events include HasSubevent, IsSubevent, Before, After, Causes, and IsResult. For the tuple [event0, relation, event1], HasSubevent indicates that event1 is a subevent of event0, IsSubevent indicates that event0 is a subevent of event1, Before indicates that event0 occurs before event1, After indicates that event0 occurs after event1, Causes indicates that event0 causes event1, and IsResult indicates that event0 is the result of event1."+"\n\n"
        # prompt = 'Answer the question using only the letter label of the option.\n\n'
        prompt += "Requirements:\nAbstract event types can only be chosen from \"Schema\", and the relationships of abstract event types can only be selected from HasSubevent, IsSubevent, Before, After, Causes, and IsResult. Follow the format in examples, output in JSONL format. The key \"event type\" should correspond to a value that is a dictionary with events as keys and their abstracted categories as values. The key \"event relation\" should correspond to a value that is a list of tuples [event0, relation, event1]. The relationships between events include HasSubevent, IsSubevent, Before, After, Causes, and IsResult.\n\n"
        
        # A = [['obstruct justice'], ['make decision'], ['alert'], ['eat breakfast', 'eat breakfast'], ['enhancing technique'], ['celebrate'], ['travel'], ['preparation material'], ['study'], ['conflict'], ['lost job'], ['shape dough', 'knead dough'], ['experience delay'], ['planting flowers'], ['purchase ingredients'], ['submit paper'], ['take break'], ['publication'], ['fly kite'], ['abandon'], ['becoming tired', 'tired', 'feel tired'], ['communication'], ['start moving'], ['shower', 'have shower'], ['get money'], ['slumber', 'sleep'], ['seek help', 'seeking assistance', 'seek assistance'], ['learn', 'learn how'], ['feeling excitement', 'excited'], ['burying cat', 'bury cat'], ['inhaling'], ['examination', 'evaluate'], ['perform'], ['play music', 'hear music'], ['agreeing with'], ['continue dancing'], ['play sport', 'play sports'], ['recover illness'], ['selecting casket', 'choosing casket'], ['exercise', 'exercising'], ['healthcare policy'], ['employment'], ['landscape gardening', 'landscaping garden'], ['conference presentation'], ['tie shoelaces'], ['engage in conversation'], ['check weather'], ['write poem', 'write poetry'], ['pursue education', 'receive education'], ['experience contentment'], ['plan trip'], ['make bread'], ['misunderstanding', 'confusion'], ['repair kite'], ['purchase snacks'], ['analyzing data', 'analyze data'], ['bake cake'], ['hearing news'], ['dry'], ['award ceremony'], ['making friends'], ['eating', 'eat'], ['playing video games'], ['community engagement'], ['passing class'], ['drink more'], ['paint artwork'], ['getting contract'], ['attend class'], ['watch nature'], ['escape consequences'], ['planning strategy'], ['prepare documentation'], ['dream'], ['create'], ['record'], ['climb tree'], ['drink refreshment'], ['academic collaboration'], ['preheat oven'], ['shop for groceries'], ['building cathedral'], ['analyse', 'analysing'], ['plead guilty'], ['digging hole'], ['read book'], ['achievement', 'accomplishment'], ['take a walk'], ['improve skill', 'improve skills'], ['swim'], ['floorspace'], ['purchase supplies'], ['hear singing'], ['observe change'], ['evening concludes'], ['submitting exam'], ['family expansion'], ['financial concerns'], ['feeling lonely'], ['reflect', 'think'], ['organize garden party'], ['follow recipe', 'following recipe'], ['repair vehicle'], ['bricks'], ['complete homework'], ['eat ice cream'], ['gaining weight'], ['lose consciousness'], ['obtain kite', 'get kite'], ['gain knowledge', 'gaining knowledge'], ['distraction'], ['decompose'], ['reacting emotionally'], ['use yeast'], ['protest begins'], ['prepare food', 'prepare meal'], ['kick feet'], ['fall'], ['common interests'], ['visit market'], ['stretching'], ['express information'], ['leave reception'], ['legal action initiation'], ['teach baking classes'], ['go outside'], ['overwhelmed'], ['play continue'], ['commit crime'], ['choose recipe'], ['depart honeymoon'], ['traveling abroad'], ['breathe'], ['daydreaming'], ['rest prepare'], ['contract termination'], ['feeling motivated', 'feel motivated'], ['doing housework'], ['attending commencement'], ['recovering from injury'], ['feel pain'], ['attend afterparty'], ['kill'], ['cooking'], ['receive feedback'], ['hunt'], ['acquiring sponsorship'], ['medication'], ['extinguishing lights'], ['cultural appreciation'], ['choose clothes'], ['legal complication'], ['cleaning'], ['have party'], ['take stand'], ['recover quickly'], ['forming relationships'], ['experience emotion'], ['get weapon'], ['drink energy drink'], ['avoid distractions'], ['revise'], ['becoming hungry'], ['celebrating success', 'celebrate success'], ['receiving financial aid'], ['study more'], ['entertainment break'], ['heal'], ['finishing projects'], ['sweating'], ['life milestone'], ['feel cold'], ['have dinner'], ['teach', 'teaching'], ['formalizing partnership'], ['publish book'], ['overlook'], ['open space'], ['destroy evidenceescape captivity'], ['social interaction'], ['provide assistance'], ['write grant proposal'], ['enjoy breeze'], ['shop'], ['miss deadline'], ['plan future'], ['argue case'], ['legislation'], ['stress relief'], ['physical endurance'], ['memorizing'], ['signing documents'], ['crying'], ['maintenance disruption'], ['withdraw money'], ['graduating'], ['consider consequences'], ['volunteering service'], ['inauguration celebration'], ['achieving academic recognition'], ['doubt', 'ignorance'], ['play'], ['economic impact'], ['feeling energized'], ['external acknowledgement'], ['schedule change'], ['inaugurate ceremony'], ['show rehabilitation'], ['get injuried'], ['brain'], ['receive applause'], ['explore'], ['happy', 'feel happy'], ['watch clouds'], ['dancing'], ['respond to questions'], ['attend seminar'], ['participate event'], ['application process'], ['expansion'], ['teach others'], ['recognize individuals'], ['pay'], ['cool down'], ['acknowledge support'], ['resolve conflict'], ['break law'], ['live'], ['improve health'], ['seek therapy'], ['compose'], ['grieve'], ['career planning'], ['run'], ['being ablebodied'], ['labour'], ['have string'], ['launch investigation'], ['seek entertainment'], ['relaxation'], ['reconciliation'], ['intensifying music'], ['get flour'], ['apathy'], ['destruction'], ['lung'], ['getting job'], ['design recipe'], ['wander'], ['reviewing news'], ['achieving perfection'], ['study recipe'], ['seek approval'], ['negotiating'], ['supporting infrastructure'], ['discuss performance'], ['close attention'], ['interact with audience'], ['pack up kite'], ['submit evidence'], ['return home'], ['physical exertion'], ['engage attention'], ['study sessions'], ['escape captivity'], ['feel discomfort'], ['plan vacation'], ['planning', 'plan'], ['become engrossed'], ['fallAsleep'], ['continuing education'], ['eat cake'], ['introduction'], ['organize materials'], ['buy'], ['move feet'], ['planning confusion'], ['pass out'], ['measure ingredients'], ['initiate project'], ['cdeact'], ['increasing skill'], ['cause disruption'], ['discuss lessons'], ['disappointed with socializing'], ['meal'], ['receive communication'], ['cogitate'], ['loss'], ['emotional support'], ['common ground'], ['read'], ['delegating tasks'], ['overlooking data'], ['sing'], ['realization'], ['escalation of disputes'], ['submit assignment'], ['understanding'], ['die'], ['understanding better'], ['dance celebration'], ['switch roles'], ['being hired'], ['seek employment'], ['prepare breakfast'], ['evaluation knowledge'], ['healthcare demand'], ['personal satisfaction'], ['decorate cake'], ['execute action'], ['isolation'], ['abolish'], ['mix ingredients', 'prepare ingredients', 'get ingredients'], ['studying'], ['exhale'], ['watch kite'], ['tie shoes'], ['accept'], ['summarize'], ['being rejected'], ['anxiety'], ['sharing knowledge'], ['receive award'], ['change of interest'], ['pack supplies'], ['landscaping'], ['financial decision'], ['renew'], ['repare oven'], ['lose balance'], ['being nice'], ['plan travel'], ['distract'], ['improving technique', 'improving techniques'], ['incite conflict'], ['rest after activity'], ['refreshment'], ['talk', 'discuss'], ['eat snack'], ['sadness'], ['listen', 'remember'], ['exhaustion'], ['use breadmaker'], ['implement policy'], ['develop skills'], ['coordinate'], ['indifference'], ['sick'], ['invent'], ['apply'], ['discover'], ['resting', 'rest'], ['trust'], ['observation'], ['implement'], ['help'], ['ponder'], ['source ingredients'], ['leave event'], ['construct'], ['learning'], ['inquiry'], ['feeling invigorated'], ['adjust kite'], ['review'], ['investigate'], ['fall asleep'], ['disinterest'], ['discuss strategy'], ['physical exercise'], ['disappointment'], ['gather information'], ['prepare equipment'], ['cook'], ['disagreement'], ['take exam'], ['prepare presentation'], ['observe weather'], ['breathing'], ['misfortune']]
        A = self.clusters
        B = list(data["event_type"].values())
        combined = find_and_sample(A, B)
        schema = ", ".join(combined)
        prompt += "Schema:\n"+schema+"\n\n"
        dem = self.build_demonstrations(list(range(self.k)))
        prompt += dem
        
        prompt += "Here are some examples:\n\n"+"Instances:\nevent53:\nThis revelation ultimately prompted an individual in the courtroom audience to discretely exit the room and later that evening, the same individual, driven by fear of exposure, went on to commit a fatal assault against a witness who could connect him to the crime.\
\nevent67:\nWhile Mr. Smith was providing his account, he mentioned a key detail that was previously overlooked—a unique tattoo that he glimpsed on the perpetrator's arm.\
\nevent91:\nTwo weeks later, during the heated court proceedings at the downtown courthouse, the homeowner, Mr. Smith, was called to testify before the jury regarding the night of the incident.\
\nevent64:\nIn a quiet suburban neighborhood, a burglary occurred at the Smith residence, where an unknown assailant broke in and stole valuable heirlooms late at night.\
\n\nContext:\n\"event64\" is before \"event91\". \"event67\" is a subevent of \"event91\".\
\nQuestion:\nWhich is the causal relationship between \"event67\" and \"event53\"?\
\nChoices:\nA. \"event67\" causes \"event53\".\
\nB. \"event67\" is result of \"event53\".\
\nC. There is no obvious causal relationship between \"event67\" and \"event53\".\
\n\nEvent type and event relation:\n{\"event_type\": {\"event67\": \"talk\", \"event53\": \"kill\", \"event64\": \"commit_crime\", \"event91\": \"take_stand\"}, \"event_relation\": [[\"commit_crime\", \"Before\", \"take_stand\"], [\"take_stand\", \"HasSubevent\", \"talk\"], [\"talk\", \"Causes\", \"kill\"]]}"+"\n\n"

        # prompt += "Examples:\n"+"""{'event_type': {'event67': 'talk', 'event53': 'kill', 'event64': 'commit_crime', 'event91': 'take_stand'}, 'event_relation': [['event64', 'Before', 'event91'], ['event91', 'HasSubevent', 'event67'], ['event67', 'Causes', 'event53']]}"""+"\n\n"
         
        prompt += "Now, based on the above, please output the event semantic knowledge used in solving the following problem.\n\n"
        prompt += formatted_string+'\n\n'
        
        choices = data['choices']
        context = data['context']
        question = data['question']
        new_question, new_choices=process_question_and_choices(question,choices)
        inds = [char for char in string.ascii_letters if char.isupper()]
        ipt = ''.join([f'{inds[i]}. {c}\n' for i, c in enumerate(new_choices)])
        lab = {}
        
        
        lab["event_type"]=data["event_type"]
        lab["event_relation"]=data["event_relation"]
        
        src=f'Context:\n{context}\nQuestion:\n{new_question}\nChoices:\n{ipt}\nEvent type and event relation:'
        prompt += src

        
        srcs = [prompt]

        yield srcs, [new_choices], json.dumps(lab)


    def __getitem__(self, index):
        src_list = self.src_list[index]
        stm_list = self.stm_list[index]
        lab = self.labs[index]
        return src_list, stm_list, lab 


    def __len__(self,):
        return len(self.src_list)

        
    def collect_fn(self, data):
        src = list(zip(*[d[0] for d in data]))
        stm = list(zip(*[d[1] for d in data]))
        labs = [d[2] for d in data]

        return {
                'src': src,
                'stm': stm,
                'labs': labs,
               }


if __name__ == '__main__':
    pass
