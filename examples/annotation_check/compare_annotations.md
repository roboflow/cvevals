## Annotations check

'annotation_check' is a script for evaluating that your local annotations match the annotations uploaded to roboflow -  defined as the pixel similarity of the 'ground truth' annotation(s) in each image compared to the equivalent robolofow annotation.

## Running the script

```bash
python run_evaluator.py --local_data_folder data/full_size_segregated --roboflow_data_folder data/intellisee_data_all-7    
```