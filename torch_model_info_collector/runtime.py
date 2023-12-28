import torch.utils.data
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def short_text():
    return "This fruit shipping company provide different vehicle options like car and"


def long_text():
    return 'Large language models (LLMs) with hundreds of billions of parameters have sparked a new wave of ' \
           'exciting AI applications. However, they are computationally expensive at inference time. Sparsity is ' \
           'a natural approach to reduce this cost, but existing methods either require costly retraining, have to ' \
           'forgo LLM’s in-context learning ability, or do not yield wall-clock time speedup on modern hardware. ' \
           'We hypothesize that contextual sparsity, which are small, input-dependent sets of attention heads and ' \
           'MLP parameters that yield approximately the same output as the dense model for a given input, ' \
           'can address these issues. We show that contextual sparsity exists, that it can be accurately predicted, ' \
           'and that we can exploit it to speed up LLM inference in wall-clock time without compromising LLM’s ' \
           'quality or in-context learning ability. Based on these insights, we propose DEJAVU, a system that uses a ' \
           'low-cost algorithm to predict contextual sparsity on the fly given inputs to each layer, along with ' \
           'an asynchronous and hardware-aware implementation that speeds up LLM inference. We validate that DEJAVU ' \
           'can reduce the inference latency of OPT-175B by over 2× compared to the state-of-the-art ' \
           'FasterTransformer, and over 6× compared to the widely used Hugging Face implementation, ' \
           'without compromising model quality. '


def long_text_2048():
    return "Once upon a time in a distant land, there lived a young wizard named Merlin. He possessed extraordinary powers and was known throughout the kingdom for his magical abilities. Merlin spent his days studying ancient tomes, delving into the secrets of the universe. His long white beard flowed down to his waist, giving him an air of wisdom and mystique."\
        "One fateful day, an urgent message arrived at Merlin's tower. The king himself requested the wizard's presence at the royal court. Intrigued by the summons, Merlin packed his belongings and embarked on a journey to the grand castle."\
        "As he arrived at the court, Merlin was greeted by the king and his advisors. They explained that a great evil had befallen the kingdom—a fearsome dragon had been terrorizing the villages, burning everything in its path. The people were living in fear, and the kingdom was in desperate need of a hero."\
        "Merlin listened intently, his mind already formulating a plan. He knew that defeating the dragon would require all his knowledge and skill. But he also knew that he could not do it alone."\
        "With the king's permission, Merlin set out to gather a band of brave knights to aid him in his quest. He sought out the finest warriors from far and wide, each with their own unique talents and strengths."\
        "Together, they trained tirelessly, honing their skills and preparing for the battle ahead. Merlin taught them ancient spells and shared his wisdom, instilling in them the courage to face the mighty dragon."\
        "Finally, the day of reckoning arrived. Merlin and his knights stood before the fearsome creature, ready to confront their greatest challenge yet. With a wave of his staff and a powerful incantation, Merlin unleashed his magic, engaging the dragon in an epic battle."\
        "The clash between magic and fire filled the air as the dragon spewed flames and Merlin countered with spells of protection. The knights fought valiantly, wielding their swords with precision and bravery. It was a battle of wills, a test of strength and determination."\
        "Hours turned into days, and days into weeks. The battle raged on, with neither side willing to give in. But slowly, steadily, Merlin and his knights gained ground. Their combined efforts weakened the dragon's defenses until, at last, they landed a final blow that sent the mighty beast crashing to the ground."\
        "The kingdom erupted in cheers as Merlin and his knights emerged victorious. The people hailed them as heroes, their saviors from the wrath of the dragon. The king rewarded them with great riches and bestowed upon Merlin the title of Royal Wizard."\
        "From that day forward, Merlin's name would be forever etched in the annals of history—a symbol of bravery, wisdom, and the power of unity. And whenever darkness threatened the land, the legend of Merlin and his knights would inspire hope and remind the world that even the greatest challenges could be overcome."\
        "As news of Merlin and his knights' triumph spread throughout the kingdom, their tale became a legend whispered among the people. The bards composed ballads that celebrated their valor and recounted their epic battles against the forces of evil. Children would gather around firesides, wide-eyed with wonder, as parents regaled them with stories of Merlin's magical prowess and the unwavering loyalty of his knights."\
        "The kingdoms neighboring Camelot heard of their exploits too, and emissaries from far and wide arrived at the court seeking alliances and guidance from the renowned Royal Wizard. Merlin, now revered not only within his own realm but also beyond its borders, embraced his new role with humility and wisdom. He became known as the advisor to kings and queens, offering counsel on matters both mundane and mystical."\
        "Under Merlin's guidance, Camelot entered an era of prosperity and enlightenment. His mastery over magic was surpassed only by his dedication to justice and fairness. The land flourished, and the people lived in harmony, grateful for the peace brought about by Merlin's influence."\
        "But even amidst the tranquil times, whispers of looming threats reached Merlin's ears. Dark forces, envious of Camelot's prosperity, plotted to bring chaos and destruction upon the kingdom. Recognizing the signs of impending danger, Merlin gathered his loyal knights once again, ready to defend the land they held dear."\
        "The battle that followed was fierce and arduous, testing the limits of their strength and resolve. Yet, united by their shared purpose, Merlin and his knights faced every challenge head-on. Their bravery inspired others to stand against the encroaching darkness, rallying the armies of good from all corners of the realm."\
        "In the end, light prevailed over shadow, and Camelot emerged victorious once more. The people rejoiced, their faith in Merlin and his knights reaffirmed. The kingdom stood as a shining beacon of hope and resilience, proof that even in the face of adversity, unity and unwavering determination could overcome any obstacle."\
        "And so, the legend of Merlin and his knights continued to grow, passing down through generations. Their story became a timeless reminder that true heroes were not born from privilege or bestowed with supernatural gifts alone but were forged through courage, compassion, and the unwavering belief in the power of good."\
        "As the years turned into centuries, the tale of Merlin and his knights remained alive, inspiring countless souls to rise above their own challenges and fight for what they believed in. And in the hearts of those who carried their legacy, the spirit of Camelot lived on, forever resonating with the enduring values of honor, justice, and the triumph of light over darkness."\
        "Centuries passed, and the legend of Merlin and his knights continued to captivate the imaginations of people across the land. Their deeds were immortalized in paintings, tapestries, and sculptures that adorned the halls of castles and cathedrals. Each retelling of their story added new layers of enchantment, further blurring the line between history and myth."\
        "As time went on, Camelot itself faded from existence, becoming a mere echo of a glorious past. The physical remnants of the once-majestic kingdom crumbled into ruins, reclaimed by nature. Yet, the spirit of Camelot endured, carried forward by those who held onto its ideals and virtues."\
        "The descendants of Merlin's knights took up the mantle, forming new orders of chivalry dedicated to upholding justice and protecting the weak. They donned armor reminiscent of their ancestors and embarked on quests to vanquish evil wherever it arose. These modern-day knights sought to spread the legacy of Camelot, ensuring that its principles would never be forgotten."\
        "Throughout the ages, individuals with extraordinary abilities emerged, claiming to be descendants of Merlin himself. Some displayed inherent magical talent, while others possessed an uncanny knack for wisdom and foresight. These individuals became known as the 'Merlinbloods', revered for their connection to the legendary wizard."\
        "The Merlinbloods formed their own council, gathering in secret to share knowledge, hone their skills, and safeguard the ancient artifacts associated with Merlin. They walked a delicate path, balancing their duty to protect humanity with the responsibility of wielding such extraordinary powers. Their existence remained hidden from the world at large, their actions veiled in mystery."\
        "Even in the modern era, whispers of Merlin\'s return circulated among those attuned to the mystical undercurrents of the world. Many believed that when darkness threatened to overshadow the light, the spirit of Merlin would manifest itself in a chosen champion. This fabled figure, the \'Once and Future Merlin,\' was said to possess the combined might of all the previous Merlins and would rise to defend the world in its darkest hour."\
        "While skeptics dismissed these tales as mere fantasy, there were those who clung to the hope that Camelot would one day be reborn. They believed that the values embodied by Merlin and his knights—courage, justice, and compassion—were timeless and could guide humanity towards a brighter future."\
        "And so, the legend of Merlin, his knights, and the realm of Camelot remained alive, carried forward through the tapestry of time. Whether as a source of inspiration, a moral compass, or a symbol of the enduring power of good, their story continued to remind people that even in the face of adversity, heroism could be found within each individual, waiting to be awakened."\
        "As the sun set on another day, casting long shadows across the land, the legacy of Camelot whispered on the wind, reminding all who listened that the spirit of Arthur, Merlin, and the Knights of the Round Table would forever dwell in the hearts of those who dared to dream of a better world."


def run_module(model_name: str, collector=None, text=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if collector is not None:
        collector.register_hook(model)

    if text is None:
        text = long_text_2048()
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, labels=inputs["input_ids"], max_length=1)

    output_text = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True, add_special_tokens=False)
    print(output_text)
    return collector


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        example = self.dataset[index]
        return example["input"]

    def __len__(self):
        return len(self.dataset)


def run_module_dataset(model_name: str, collector=None, dataset_name_subset_split=None, batch_size=16):
    dataset_name, dataset_subset, dataset_split = dataset_name_subset_split
    dataset = load_dataset(dataset_name, dataset_subset, split=dataset_split)
    dataset = dataset.filter(lambda x: len(x['text']) > 100).map(lambda x: {'text': x['text'][:2048]})
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if collector is not None:
        collector.register_hook(model)

    for idx, batch in enumerate(dataloader):
        if idx > 0:
            break
        inputs = tokenizer(batch['text'], return_tensors="pt", padding=True)
        outputs = model.generate(**inputs, labels=inputs["input_ids"], max_length=1)
        _ = outputs
        # output_text = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True, add_special_tokens=False)
        # print(output_text)

    return collector
