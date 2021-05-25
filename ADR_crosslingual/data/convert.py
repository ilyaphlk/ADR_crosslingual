from web_anno_tsv import open_web_anno_tsv
from glob import glob
import os


def webanno2brat(webanno_dir, out_dir):
    os.mkdir(out_dir)
    os.mkdir(os.path.join(out_dir, "text"))
    os.mkdir(os.path.join(out_dir, "annotation"))
    skipped_counter = 0
    for tsv_file in glob(os.path.join(webanno_dir, 'annotation/*/*.tsv')):
        filename = tsv_file.split('/')[-2]
        try:
            with open_web_anno_tsv(tsv_file) as f:
                accumulated_offset = 0
                entity_id = 1
                sentences = []
                str_entities = []
                for i, sentence in enumerate(f):
                    sentences.append(sentence.text)
                    for j, annotation in enumerate(sentence.annotations):
                        str_entity = ("T"+str(entity_id)+'\t'+annotation.label+" "
                                      +str(accumulated_offset + annotation.start)+" "
                                      +str(accumulated_offset + annotation.stop)+'\t'+annotation.text)
                        str_entities.append(str_entity)
                        entity_id += 1

                    accumulated_offset += len(sentence.text) + 1

                text = '\n'.join(sentences) + '\n'
                entities = '\n'.join(str_entities) + '\n'
                
                txt_path = os.path.join(out_dir, "text", filename[:-4]+'.txt')
                with open(txt_path, 'w+') as txt_file:
                    txt_file.write(text)

                ann_path = os.path.join(out_dir, "annotation", filename[:-4]+'.ann')
                with open(ann_path, 'w+') as ann_file:
                    ann_file.write(entities)

        except:
            skipped_counter += 1
            print(f"skipping file: {tsv_file}")


    print(f"Number of skipped files: {skipped_counter}")
