import json
import random
from copy import deepcopy
from typing import List, Dict

import numpy as np
from openai import OpenAI
from tqdm.auto import tqdm


class DataAugmentor:
    
    def __init__(self, args) -> None:
        self.args = args
        
        self.model_name = args.model_name
        if args.api_key == "EMPTY" and args.base_url == "EMPTY":
            self.client = OpenAI()
        else:
            self.client = OpenAI(
                api_key=args.api_key,
                base_url=args.base_url
            )
        self.err_cnt = 0

    @staticmethod
    def format_reasoning_text(reasoning_steps: List[str]) -> str:
        if len(reasoning_steps) == 0:
            return ""

        return "\n".join(reasoning_steps) + "\n\n"

    @staticmethod
    def split_reasoning_chains(reasoning_steps: List[str]) -> List[List[str]]:
        if len(reasoning_steps) == 0:
            return []

        chain_blocks = []
        current_block = []
        for step in reasoning_steps:
            current_block.append(step)
            if step.startswith("conclusion: "):
                chain_blocks.append(current_block)
                current_block = []

        if len(current_block) > 0:
            chain_blocks.append(current_block)

        return chain_blocks

    def build_adv_reasoning_text(self, reasoning_steps: List[str], item: Dict, name_list: List) -> str:
        chain_blocks = self.split_reasoning_chains(reasoning_steps)
        cot_noise_blocks = self.get_cot_noise(
            item=item,
            reasoning_steps=reasoning_steps,
            name_list=name_list,
            cot_noise1=getattr(self.args, "cot_noise1", 0),
            cot_noise2=getattr(self.args, "cot_noise2", 0),
        )

        if len(chain_blocks) <= 1 or len(cot_noise_blocks) == 0:
            return self.format_reasoning_text(deepcopy(reasoning_steps))

        slot_num = len(chain_blocks) - 1
        slot_to_noise_blocks = [[] for _ in range(slot_num)]
        for noise_block in cot_noise_blocks:
            slot_idx = random.randint(0, slot_num - 1)
            slot_to_noise_blocks[slot_idx].append(noise_block)

        reasoning_lines = []
        for block_idx, chain_block in enumerate(chain_blocks):
            reasoning_lines.extend(chain_block)
            if block_idx < slot_num:
                for noise_block in slot_to_noise_blocks[block_idx]:
                    reasoning_lines.extend(noise_block)

        return self.format_reasoning_text(reasoning_lines)

    def get_cot_noise(self, item: Dict, reasoning_steps: List[str], name_list: List, cot_noise1: float, cot_noise2: float) -> List[List[str]]:
        cot_noise_blocks = []
        if len(reasoning_steps) == 0:
            return cot_noise_blocks

        cot_noise1 = min(max(cot_noise1, 0), 1)
        if cot_noise1 > 0:
            cot_noise1_blocks = self.build_cot_noise1_blocks(
                reasoning_steps=reasoning_steps,
                original_name=item['name'],
                name_list=name_list,
                cot_noise1_rate=cot_noise1,
            )
            cot_noise_blocks.extend(cot_noise1_blocks)

        cot_noise2 = min(max(cot_noise2, 0), 1)
        if cot_noise2 > 0:
            cot_noise2_num = round(len(reasoning_steps) * cot_noise2)
            if cot_noise2_num > 0:
                cot_noise2_blocks = self.build_cot_noise2_blocks(item=item, block_num=cot_noise2_num)
                cot_noise_blocks.extend(cot_noise2_blocks)

        return cot_noise_blocks

    def build_cot_noise1_blocks(self, reasoning_steps: List[str], original_name: str, name_list: List, cot_noise1_rate: float) -> List[List[str]]:
        if len(name_list) == 0:
            return []

        chain_blocks = self.split_reasoning_chains(reasoning_steps)
        sampled_name_list = random.sample(name_list, min(3, len(name_list)))
        candidate_blocks = []

        for chain_block in chain_blocks:
            if not any(original_name in step for step in chain_block):
                continue

            selected_name = random.sample(sampled_name_list, 1)[0]
            while selected_name == original_name:
                selected_name = random.sample(sampled_name_list, 1)[0]

            replaced_block = []
            for step in chain_block:
                replaced_block.append(step.replace(original_name, selected_name))

            candidate_blocks.append(replaced_block)

        cot_noise1_num = int(np.ceil(len(candidate_blocks) * cot_noise1_rate))
        if len(candidate_blocks) == 0:
            return []

        cot_noise1_num = max(1, cot_noise1_num)
        cot_noise1_num = min(len(candidate_blocks), cot_noise1_num)

        return random.sample(candidate_blocks, cot_noise1_num)

    def build_cot_noise2_blocks(self, item: Dict, block_num: int) -> List[List[str]]:
        distracting_facts = deepcopy(item['distracting_facts'])
        distracting_rules = deepcopy(item['distracting_rules'])
        allow_rule_repeats = getattr(self.args, "cot_noise2_allow_repeats", False)
        if len(distracting_facts) == 0 and len(distracting_rules) == 0:
            return []

        cot_noise2_blocks = []
        remaining_rules = deepcopy(distracting_rules)
        for _ in range(block_num):
            block = []

            max_fact_num = min(2, len(distracting_facts))
            sampled_fact_num = random.randint(1, max_fact_num) if max_fact_num > 0 else 0
            sampled_facts = self.sample_items(distracting_facts, sampled_fact_num)
            for fact_idx, fact in enumerate(sampled_facts):
                block.append(f"fact{fact_idx + 1}: {fact}")

            if len(remaining_rules) > 0:
                sampled_rule = random.sample(remaining_rules, 1)[0]
                remaining_rules.remove(sampled_rule)
            elif len(distracting_rules) > 0 and allow_rule_repeats:
                sampled_rule = random.sample(distracting_rules, 1)[0]
            else:
                sampled_rule = None

            if sampled_rule is None and len(sampled_facts) == 0:
                break

            if sampled_rule is not None:
                block.append(f"rule: {sampled_rule}")

            conclusion = self.build_noise2_conclusion(sampled_rule=sampled_rule, sampled_facts=sampled_facts)
            if conclusion is not None:
                block.append(f"conclusion: {conclusion}")

            if len(block) > 0:
                cot_noise2_blocks.append(block)

        return cot_noise2_blocks

    @staticmethod
    def build_noise2_conclusion(sampled_rule: str, sampled_facts: List[str]) -> str:
        if sampled_rule is not None:
            lowered_rule = sampled_rule.lower()
            if " then " in lowered_rule:
                split_idx = lowered_rule.index(" then ")
                prefix_len = split_idx + len(" then ")
                conclusion = sampled_rule[prefix_len:].strip()
            else:
                conclusion = sampled_rule.strip()

            if conclusion.endswith("."):
                conclusion = conclusion[:-1]
            if len(conclusion) > 0:
                conclusion = conclusion[0].upper() + conclusion[1:] + "."
                return conclusion

        if len(sampled_facts) > 0:
            return sampled_facts[-1]

        return None

    @staticmethod
    def sample_items(candidates: List[str], sample_num: int) -> List[str]:
        if sample_num <= 0 or len(candidates) == 0:
            return []

        if sample_num <= len(candidates):
            return random.sample(candidates, sample_num)

        sampled_items = deepcopy(candidates)
        remaining_num = sample_num - len(candidates)
        sampled_items.extend(random.choices(candidates, k=remaining_num))
        random.shuffle(sampled_items)
        return sampled_items
        
    def step_augment(self, data: List, shuffled: bool, has_noise1: float, has_noise2: float, name_list: List, start: int, end: int) -> List:
        """
        1. Introduce noise
        2. Generate reasoning chain and then create problems at each step
        3. Return the result
        """
        unmatched_context = 0
        current_dataset = []
        err_cnt = 0
        
        cnt = -1
        pbar = tqdm(data)
        for item in pbar:
            cnt += 1
            if cnt < start or cnt >= end:
                continue
            
            # symbolic representation
            nl2fol = {}
            for i in range(len(item['facts'])):
                nl2fol[item['facts'][i]] = item['facts_fol'][i]
            for i in range(len(item['rules'])):
                nl2fol[item['rules'][i]] = item['rules_fol'][i]
            for i in range(len(item['distracting_facts'])):
                nl2fol[item['distracting_facts'][i]] = item['distracting_facts_fol'][i]
            for i in range(len(item['distracting_rules'])):
                nl2fol[item['distracting_rules'][i]] = item['distracting_rules_fol'][i]
            
            base_context = deepcopy(item['context'])
            err_flag = False
            
            # 1: introduce noise
            noise_list = self.get_noise(item=item, base_context=base_context, name_list=name_list, has_noise1=has_noise1, has_noise2=has_noise2)

            # augment problems
            accumulated_context = []
            accumulated_reasoning_steps = []
            
            # problem quality control
            check_context = []
            for chain in item['reasoning_chains']:
                for fact in chain['facts']:
                    if fact in base_context:
                        check_context.append(fact)
                
                if chain['rules'] is not None:
                    check_context.append(chain['rules'])
            
            try:
                assert set(base_context) == set(check_context)
            except:
                unmatched_context += 1
                continue
            
            # 2: generate reasoning chains
            for chain_idx in range(len(item['reasoning_chains'])):
                current_problem = {'id': len(current_dataset)}
                
                chain = item['reasoning_chains'][chain_idx]
                chain_fol = item['reasoning_chains_fol'][chain_idx]
                if chain['conclusion'] is None or chain == item['reasoning_chains'][-1]:  # in step augmentation, we ignore the last step
                    err_flag = True
                    break
                else:
                    # get facts
                    for i in range(len(chain['facts'])):
                        if item['name'].lower() in chain_fol['facts'][i].lower():
                            if item['name'] not in chain['facts'][i]:  # quality control
                                err_cnt += 1
                                err_flag = True
                                break
                        
                        accumulated_reasoning_steps.append(f"fact{i + 1}: {chain['facts'][i]}")
                        if chain['facts'][i] in base_context:
                            accumulated_context.append(chain['facts'][i])
                    
                    # get rules
                    if chain['rules'] is not None:
                        if item['name'].lower() in chain_fol['rules'].lower():  # quality control
                            if item['name'] not in chain['rules']:
                                err_cnt += 1
                                err_flag = True
                                break
                    
                        accumulated_reasoning_steps.append(f"rule: {chain['rules']}")
                        accumulated_context.append(chain['rules'])
                        
                        # conclusion
                        if item['name'].lower() in chain_fol['conclusion'].lower():
                            if item['name'] not in chain['conclusion']:
                                err_cnt += 1
                                err_flag = True
                                break
                        
                        accumulated_reasoning_steps.append(f"conclusion: {chain['conclusion']}")
                    else:
                        assert chain == item['reasoning_chains'][-1]
                    
                    current_problem['options'] = random.sample([["A) True", "B) False"], ["A) True", "B) False", "C) Uncertain"]], 1)[0]
                    current_question = "Based on the above information, is the following statement true, false, or uncertain? " if current_problem['options'] == ["A) True", "B) False", "C) Uncertain"] else "Based on the above information, is the following statement true, or false? "
                    
                    current_anwer = random.sample(['True', 'False'], 1)[0]
                    current_problem['answer'] = 'A' if current_anwer == "True" else "B"
                    
                    if current_problem['answer'] == 'A':
                        current_question += chain['conclusion']
                        clean_prefix = self.format_reasoning_text(accumulated_reasoning_steps)
                        adv_prefix = self.build_adv_reasoning_text(accumulated_reasoning_steps, item, name_list)
                        current_problem['reasoning'] = clean_prefix + f"Therefore, it is true that {chain['conclusion']} The correct option is: A."
                        current_problem['adv_reasoning'] = adv_prefix + f"Therefore, it is true that {chain['conclusion']} The correct option is: A."
                    else:
                        # get the negated conclusion
                        negation_message = [
                            {'role': 'system', 'content': "You are a language expert skilled in transforming sentences into their negative forms. Your answer should in JSON format with key: negation"},
                            {'role': 'user', 'content': "Sapphire is conventional."},
                            {'role': 'assistant', 'content': "{\n  \"negation\": \"Sapphire is not conventional.\"\n}"},
                            {'role': 'user', 'content': "Brantley does not conduct research."},
                            {'role': 'assistant', 'content': "{\n  \"negation\": \"Brantley conducts research.\"\n}"},
                            {'role': 'user', 'content': chain['conclusion']}
                        ]
                        negated_conclusion = self.send_request(messages=negation_message, key='negation')
                        current_question += negated_conclusion
                        clean_prefix = self.format_reasoning_text(accumulated_reasoning_steps)
                        adv_prefix = self.build_adv_reasoning_text(accumulated_reasoning_steps, item, name_list)
                        current_problem['reasoning'] = clean_prefix + f"Therefore, it is false that {negated_conclusion} The correct option is: B."
                        current_problem['adv_reasoning'] = adv_prefix + f"Therefore, it is false that {negated_conclusion} The correct option is: B."
                        
                    current_problem['question'] = current_question
                
                # keep the clean context and store the perturbed version separately
                clean_context = deepcopy(accumulated_context)
                adv_context = deepcopy(accumulated_context)

                remaining_context = []  # the remaining context can be used as distractions
                for b_context in base_context:
                    if b_context not in accumulated_context:
                        remaining_context.append(b_context)

                if len(remaining_context) != 0:
                    sampled_base_context_num = random.sample(range(len(remaining_context)), 1)[0]
                    adv_context.extend(remaining_context[:sampled_base_context_num])
                
                if len(noise_list) != 0:
                    sampled_noise_num = random.randint(1, len(noise_list))
                    adv_context.extend(noise_list[:sampled_noise_num])
                
                if shuffled:
                    random.shuffle(adv_context)

                current_problem['context'] = " ".join(clean_context)
                current_problem['adv_context'] = " ".join(adv_context)
                
                # symbolic representation, background story and keywords
                current_problem['nl2fol'] = deepcopy(nl2fol)
                current_problem['background_story'] = deepcopy(item['background_story'])
                current_problem['name'] = deepcopy(item['name'])
                current_problem['keyword'] = deepcopy(item['keyword'])
                current_problem['subject_category'] = deepcopy(item['subject_category'])
                
                current_dataset.append(current_problem)
            
            if err_flag:
                continue
        
        print(unmatched_context)
        return current_dataset
    
    def uncertain_augment(self, data: List, shuffled: bool, has_noise1: float, has_noise2: float, name_list: List, start: int, end: int) -> List:
        """
        1. Load word list and introduce noise
        2. Generate reasoning chain and then create problems at each step
        3. Return the result
        """
        unmatched_context = 0
        current_dataset = []
        
        # load word list
        with open(self.args.predicate_file, 'r') as f:
            word_list = json.load(f)['words']
        
        cnt = -1
        pbar = tqdm(data)
        for item in pbar:
            cnt += 1
            if cnt < start or cnt >= end:
                continue
            
            # symbolic representation
            nl2fol = {}
            for i in range(len(item['facts'])):
                nl2fol[item['facts'][i]] = item['facts_fol'][i]
            for i in range(len(item['rules'])):
                nl2fol[item['rules'][i]] = item['rules_fol'][i]
            for i in range(len(item['distracting_facts'])):
                nl2fol[item['distracting_facts'][i]] = item['distracting_facts_fol'][i]
            for i in range(len(item['distracting_rules'])):
                nl2fol[item['distracting_rules'][i]] = item['distracting_rules_fol'][i]
                
            base_context = deepcopy(item['context'])
            err_flag = False
            
            noise_list = self.get_noise(item=item, base_context=base_context, name_list=name_list, has_noise1=has_noise1, has_noise2=has_noise2)
            
            # start augmenting
            accumulated_context = []
            accumulated_reasoning_steps = []
            
            # quality control
            check_context = []
            for chain in item['reasoning_chains']:
                for fact in chain['facts']:
                    if fact in base_context:
                        check_context.append(fact)

                if chain['rules'] is not None:
                    check_context.append(chain['rules'])
            
            try:
                assert set(base_context) == set(check_context)
            except:
                unmatched_context += 1
                continue
            
            # Break down each step of reasoning to generate a new problem with uncertain answer
            for chain_idx in range(len(item['reasoning_chains'])):
                current_problem = {'id': len(current_dataset)}
                
                chain = item['reasoning_chains'][chain_idx]
                chain_fol = item['reasoning_chains_fol'][chain_idx]
                
                if chain['conclusion'] is None or chain == item['reasoning_chains'][-1]:
                    break  # uncertain problems is not needed when performing uncertainty augmentation
                else:
                    current_problem['answer'] = "C"
                    current_problem['options'] = ["A) True", "B) False", "C) Uncertain"]
                    
                    # get facts
                    for i in range(len(chain['facts'])):
                        if item['name'].lower() in chain_fol['facts'][i].lower():
                            if item['name'] not in chain['facts'][i]:
                                err_flag = True
                                break
                            
                        accumulated_reasoning_steps.append(f"fact{i + 1}: {chain['facts'][i]}")
                        if chain['facts'][i] in base_context:
                            accumulated_context.append(chain['facts'][i])
                    
                    # get rules
                    if chain['rules'] is None:
                        err_flag = True
                        break
                        
                    if item['name'].lower() in chain_fol['rules'].lower():
                        if item['name'] not in chain['rules']:
                            err_flag = True
                            break
                    
                    accumulated_reasoning_steps.append(f"rule: {chain['rules']}")
                    accumulated_context.append(chain['rules'])
                    
                    # conclusion
                    if item['name'].lower() in chain_fol['conclusion'].lower():
                        if item['name'] not in chain['conclusion']:
                            err_flag = True
                            break
                    accumulated_reasoning_steps.append(f"conclusion: {chain['conclusion']}")
                    
                    # there are two types of uncertain conclusions: replaced subject and completely unrelated facts or rules
                    uncertain_type = np.random.choice(
                        a=np.array([0, 1]),
                        size=1,
                        replace=True,
                        p=[0.7, 0.3]
                    )
                    if item['name'] not in chain['conclusion']:
                        uncertain_type = 1
                        print("Found Universal Conclusions")
                        
                    if uncertain_type == 0:  # replaced names
                        noise_name = random.sample(name_list, 1)[0]
                        while noise_name == item['name']:
                            noise_name = random.sample(name_list, 1)[0]
                        noise_conclusion = chain['conclusion'].replace(item['name'], noise_name)
                        
                        current_question = "Based on the above information, is the following statement true, false, or uncertain? "
                        current_question += noise_conclusion
                        current_problem['question'] = current_question

                        clean_prefix = self.format_reasoning_text(accumulated_reasoning_steps)
                        adv_prefix = self.build_adv_reasoning_text(accumulated_reasoning_steps, item, name_list)
                        current_problem['reasoning'] = clean_prefix + f"According to the context and the conclusions that have already been drawn, we can deduce that it is true that {chain['conclusion']} But it is uncertain that {noise_conclusion} The correct option is: C."
                        current_problem['adv_reasoning'] = adv_prefix + f"According to the context and the conclusions that have already been drawn, we can deduce that it is true that {chain['conclusion']} But it is uncertain that {noise_conclusion} The correct option is: C."
                    else:  # completely unrelated facts or rules
                        unrelated_message = [
                            {'role': 'system', 'content': "You are a language expert skilled in transforming a keyword and name into a statement about a fact or a commonse rule. Your answer should be simple and natural with no more than 10 words. Your answer should in JSON format with key: statement"},
                            {'role': 'user', 'content': "keyword: conventional\nname: Sapphire"},
                            {'role': 'assistant', 'content': "{\n  \"statement\": \"Sapphire is a conventional person.\"\n}"},
                            {'role': 'user', 'content': "keyword: research\nname: Brantley"},
                            {'role': 'assistant', 'content': "{\n  \"statement\": \"If Brantley conducts research, then he is a researcher or a student.\"\n}"},
                            {'role': 'user', 'content': f"keyword: {random.sample(word_list, 1)[0]}\nname: {item['name']}"}
                        ]
                        unrelated_conclusion = self.send_request(unrelated_message, key='statement')    
                        current_question = "Based on the above information, is the following statement true, false, or uncertain? "
                        current_question += unrelated_conclusion
                        current_problem['question'] = current_question
                        
                        clean_prefix = self.format_reasoning_text(accumulated_reasoning_steps)
                        adv_prefix = self.build_adv_reasoning_text(accumulated_reasoning_steps, item, name_list)
                        current_problem['reasoning'] = clean_prefix + f"According to the context and the conclusions that have already been drawn, we can deduce that it is uncertain that {unrelated_conclusion} The correct option is: C."
                        current_problem['adv_reasoning'] = adv_prefix + f"According to the context and the conclusions that have already been drawn, we can deduce that it is uncertain that {unrelated_conclusion} The correct option is: C."
                    
                    # keep the clean context and store the perturbed version separately
                    clean_context = deepcopy(accumulated_context)
                    adv_context = deepcopy(accumulated_context)

                    remaining_context = []
                    for b_context in base_context:
                        if b_context not in accumulated_context:
                            remaining_context.append(b_context)
                            
                    if len(remaining_context) != 0:  # The premises that were not selected can be used as a distraction.
                        sampled_base_context_num = random.sample(range(len(remaining_context)), 1)[0]
                        adv_context.extend(remaining_context[:sampled_base_context_num])
                    
                    if len(noise_list) != 0:
                        sampled_noise_num = random.randint(1, len(noise_list))
                        adv_context.extend(noise_list[:sampled_noise_num])
                        
                    if shuffled:
                        random.shuffle(adv_context)

                    current_problem['context'] = " ".join(clean_context)
                    current_problem['adv_context'] = " ".join(adv_context)
                    
                    # symbolic representation, background story and keywords
                    current_problem['nl2fol'] = deepcopy(nl2fol)
                    current_problem['background_story'] = deepcopy(item['background_story'])
                    current_problem['name'] = deepcopy(item['name'])
                    current_problem['keyword'] = deepcopy(item['keyword'])
                    current_problem['subject_category'] = deepcopy(item['subject_category'])
                
                    current_dataset.append(current_problem)
                    
            if err_flag:
                continue
            
        return current_dataset
                    
    def normal_generation(self, data: List, shuffled: bool, has_noise1: float, has_noise2: float, name_list: List, start: int, end: int) -> List:
        """
        1. Introduce noise
        2. Generate reasoning chain and then create problems
        3. Return the result
        """
        current_dataset = []
        err_cnt = 0
        
        # creating problems
        cnt = -1
        pbar = tqdm(data)
        for item in pbar:
            cnt += 1
            if cnt < start or cnt >= end:
                continue
            
            base_context = deepcopy(item['context'])
            err_flag = False
            
            noise_list = self.get_noise(item=item, base_context=base_context, name_list=name_list, has_noise1=has_noise1, has_noise2=has_noise2)
            
            # create problems
            context = []
            reasoning_steps = []
            current_problem = {'id': len(current_dataset)}
            
            for chain_idx in range(len(item['reasoning_chains'])):
                chain = item['reasoning_chains'][chain_idx]
                chain_fol = item['reasoning_chains_fol'][chain_idx]
                if chain['conclusion'] is None:
                    if chain == item['reasoning_chains'][-1]:
                        current_problem['options'] = ["A) True", "B) False", "C) Uncertain"]
                        assert item['answer'] == "Uncertain"
                        current_problem['answer'] = "C"
                        
                        current_question = "Based on the above information, is the following statement true, false, or uncertain? "
                        current_question += item['conclusion']
                        current_problem['question'] = current_question
                        
                        clean_prefix = self.format_reasoning_text(reasoning_steps)
                        adv_prefix = self.build_adv_reasoning_text(reasoning_steps, item, name_list)
                        current_problem['reasoning'] = clean_prefix + f"According to the context and the conclusions that have already been drawn, we can deduce that it is uncertain that {item['conclusion']} The correct option is: C."
                        current_problem['adv_reasoning'] = adv_prefix + f"According to the context and the conclusions that have already been drawn, we can deduce that it is uncertain that {item['conclusion']} The correct option is: C."
                    else:
                        continue
                else:
                    # facts
                    for i in range(len(chain['facts'])):
                        if item['name'].lower() in chain_fol['facts'][i].lower():
                            if item['name'] not in chain['facts'][i]:
                                err_cnt += 1
                                err_flag = True
                                break
                        
                        reasoning_steps.append(f"fact{i + 1}: {chain['facts'][i]}")
                        if chain['facts'][i] in base_context:
                            context.append(chain['facts'][i])

                    # rules
                    if chain['rules'] is None:
                        err_cnt += 1
                        err_flag = True
                        break
                    
                    if item['name'].lower() in chain_fol['rules'].lower():
                        if item['name'] not in chain['rules']:
                            err_cnt += 1
                            err_flag = True
                            break

                    reasoning_steps.append(f"rule: {chain['rules']}")
                    context.append(chain['rules'])
                    
                    # conclusion
                    if item['name'].lower() in chain_fol['conclusion'].lower():
                        if item['name'] not in chain['conclusion']:
                            err_cnt += 1
                            err_flag = True
                            break
                        
                    current_problem['options'] = ["A) True", "B) False", "C) Uncertain"]
                    
                    if chain == item['reasoning_chains'][-1]:
                        current_anwer = item['answer']
                        assert current_anwer != 'Uncertain'
                        
                        current_problem['answer'] = 'A' if current_anwer == "True" else "B"
                        
                        if current_anwer == "False":
                            negation_message = [
                                {'role': 'system', 'content': "You are a language expert skilled in transforming sentences into their negative forms. Your answer should in JSON format with key: negation"},
                                {'role': 'user', 'content': "Sapphire is conventional."},
                                {'role': 'assistant', 'content': "{\n  \"negation\": \"Sapphire is not conventional.\"\n}"},
                                {'role': 'user', 'content': "Brantley does not conduct research."},
                                {'role': 'assistant', 'content': "{\n  \"negation\": \"Brantley conducts research.\"\n}"},
                                {'role': 'user', 'content': chain['conclusion']}
                            ]
                            negated_conclusion = self.send_request(messages=negation_message, key='negation')
                            negated_conclusion = chain['conclusion']
                            reasoning_steps.append(f"conclusion: {negated_conclusion}")
                        else:
                            reasoning_steps.append(f"conclusion: {chain['conclusion']}")
                        
                        current_question = "Based on the above information, is the following statement true, false, or uncertain? "
                        current_question += chain['conclusion']
                        current_problem['question'] = current_question
                        
                        clean_prefix = self.format_reasoning_text(reasoning_steps)
                        adv_prefix = self.build_adv_reasoning_text(reasoning_steps, item, name_list)
                        current_problem['reasoning'] = clean_prefix + f"Therefore, it is {current_anwer.lower()} that {chain['conclusion']} The correct option is: {current_problem['answer']}."
                        current_problem['adv_reasoning'] = adv_prefix + f"Therefore, it is {current_anwer.lower()} that {chain['conclusion']} The correct option is: {current_problem['answer']}."
                    else:
                        reasoning_steps.append(f"conclusion: {chain['conclusion']}")
                        
            if err_flag:
                continue 
                
            # keep the clean context and store the perturbed version separately
            clean_context = deepcopy(context)
            adv_context = deepcopy(context)
            
            if len(noise_list) != 0:
                sampled_noise_num = random.randint(1, len(noise_list))
                adv_context.extend(noise_list[:sampled_noise_num])
                
            if shuffled:  # shuffl context
                random.shuffle(adv_context)

            current_problem['context'] = " ".join(clean_context)
            current_problem['adv_context'] = " ".join(adv_context)
            
            # symbolic representation
            nl2fol = {}
            for i in range(len(item['facts'])):
                nl2fol[item['facts'][i]] = item['facts_fol'][i]
            for i in range(len(item['rules'])):
                nl2fol[item['rules'][i]] = item['rules_fol'][i]
            for i in range(len(item['distracting_facts'])):
                nl2fol[item['distracting_facts'][i]] = item['distracting_facts_fol'][i]
            for i in range(len(item['distracting_rules'])):
                nl2fol[item['distracting_rules'][i]] = item['distracting_rules_fol'][i]
            
            current_problem['nl2fol'] = nl2fol
            
            # background story and keywords
            current_problem['background_story'] = item['background_story']
            current_problem['name'] = item['name']
            current_problem['keyword'] = item['keyword']
            current_problem['subject_category'] = item['subject_category']
 
            current_dataset.append(current_problem)
            
        return current_dataset
    
    def get_noise(self, item: Dict, base_context: List, name_list: List, has_noise1: float, has_noise2: float) -> List:
        """ Introduce distractions """
        # get distraction 2
        d_premises_list = deepcopy(item['distracting_facts'])
        d_premises_list.extend(deepcopy(item['distracting_rules']))
        
        if has_noise2 == 1:
            noise2_rate = self.sample_rate()
            noise2_num = round(len(d_premises_list) * noise2_rate)
            noise2_list = random.sample(d_premises_list, noise2_num)
        else:
            noise2_list = []
                
        # get distraction 1
        noise1_rate = min(max(has_noise1, 0), 1)
        if noise1_rate > 0:
            noise1_raw_list = deepcopy(base_context)
            random.shuffle(noise1_raw_list)
            noise1_candidates = []
                
            sampled_name_list = random.sample(name_list, min(3, len(name_list)))
                
            for i in range(len(noise1_raw_list)):
                if item['name'] in noise1_raw_list[i]:
                    selected_name = random.sample(sampled_name_list, 1)[0]
                    while selected_name == item['name']:
                        selected_name = random.sample(sampled_name_list, 1)[0]
                        
                    noise1_candidates.append(noise1_raw_list[i].replace(item['name'], selected_name))

            noise1_num = int(np.ceil(len(noise1_candidates) * noise1_rate))
            if len(noise1_candidates) > 0:
                noise1_num = max(1, noise1_num)
                noise1_num = min(len(noise1_candidates), noise1_num)
                noise1_list = random.sample(noise1_candidates, noise1_num)
            else:
                noise1_list = []
        else:
            noise1_list = []
                
        noise_list = deepcopy(noise1_list)
        noise_list.extend(deepcopy(noise2_list))
            
        return noise_list
        
    def send_request(self, messages: List, key: str) -> str:
        while True:
            api_flag = False
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.6
                )
                answer_str = completion.choices[0].message.content.replace("```json", "").replace("```", "")
                result = eval(answer_str)[key]
                api_flag = True
            except:
                self.err_cnt += 1
                print(f"API error occured, wait for 2 seconds. Error count: {self.err_cnt}")
            
            if api_flag:
                break
        
        return result
            
    @staticmethod
    def sample_rate():
        mean = 0.5
        std_dev = 0.16
        sampled_rate = np.random.normal(loc=mean, scale=std_dev, size=1)[0]
    
        if sampled_rate > 1:
            sampled_rate = 1
        elif sampled_rate < 0:
            sampled_rate = 0
        else:
            sampled_rate = sampled_rate
        
        return sampled_rate
    
    
    
    
    
    
    
