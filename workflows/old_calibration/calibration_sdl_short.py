# calibration_sdl_short.py
from calibration_sdl_base import *
import sys
sys.path.append("../utoronto_demo")
import os
import logging
from master_usdl_coordinator import Lash_E
import recommenders.pipeting_optimizer_v2 as recommender
import slack_agent
from datetime import datetime
import recommenders.llm_optimizer as llm_opt
LLM_AVAILABLE = True


# --- Experiment Config ---
LIQUID = "glycerol"  #<------------------- CHANGE THIS!
SIMULATE = False #<--------- CHANGE THIS!

DENSITY_LIQUID = LIQUIDS[LIQUID]["density"]
NEW_PIPET_EACH_TIME_SET = LIQUIDS[LIQUID]["refill_pipets"]

SEED = 7
SOBOL_CYCLES_PER_VOLUME = 2
BAYES_CYCLES_PER_VOLUME = 0
REPLICATES = 3
BATCH_SIZE = 3  # Number of suggestions per cycle (unified for both LLM and Bayesian)
VOLUMES = [0.05] #If time try different volumes! Eg 0.01 0.02 0.1
#MODELS = ['qEI', 'qLogEI', 'qNEHVI, 'LLM]
MODELS = ['LLM'] #Change this!
USE_EXISTING_DATA = True
EXISTING_DATA_FOLDER = r"C:\Users\Imaging Controller\Desktop\Calibration_SDL_Output\autosave_calibration\0922_8params"

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/calibration_vials_short.csv"
EXPECTED_MASSES = [v * DENSITY_LIQUID for v in VOLUMES]
EXPECTED_TIME = [v * 10.146 + 9.5813 for v in VOLUMES]
EXPECTED_ABSORBANCE = []

state = {
    "measurement_vial_index": 0,
    "measurement_vial_name": "measurement_vial_0"
}

# --- Init ---
lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE, initialize_biotek=False)
lash_e.nr_robot.check_input_file()
lash_e.nr_robot.move_vial_to_location("measurement_vial_0", "clamp", 0)

lash_e.logger.info("Liquid: ", LIQUIDS[LIQUID])
if not SIMULATE:
    slack_agent.send_slack_message(f"Starting new calibration experiment with {LIQUID} and models {MODELS}")

for model_type in MODELS:
    if not SIMULATE:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S" + f"_{LIQUID}"+f"_{model_type}")
        base_autosave_dir = r"C:\Users\Imaging Controller\Desktop\Calibration_SDL_Output\autosave_calibration"
        autosave_dir = os.path.join(base_autosave_dir, timestamp)
        os.makedirs(autosave_dir, exist_ok=True)
        autosave_summary_path = os.path.join(autosave_dir, "experiment_summary_autosave.csv")
        autosave_raw_path = os.path.join(autosave_dir, "raw_replicate_data_autosave.csv")
    else:
        autosave_raw_path=None
        autosave_summary_path=None

    # --- Optimization Loop ---
    ax_client = recommender.create_model(SEED, SOBOL_CYCLES_PER_VOLUME * len(VOLUMES), bayesian_batch_size=BATCH_SIZE, volume=VOLUMES, model_type=model_type, simulate=SIMULATE)
    
    # ...existing code...
    # Store existing data for LLM access
    existing_data_for_llm = []
    if USE_EXISTING_DATA:
        base_folder = EXISTING_DATA_FOLDER
        #base_folder = r"C:\Users\owenm\OneDrive\Desktop\Calibration_SDL"
        additional_folders = [folder for folder in os.listdir(base_folder) if "glycerol" in folder.lower() and os.path.isdir(os.path.join(base_folder, folder))]
        # ...existing code...
        for folder in additional_folders:
            file_path = os.path.join(base_folder, folder, "experiment_summary.csv")
            if os.path.exists(file_path):
                lash_e.logger.info(f"Loading existing data from {file_path}")
                recommender.load_data(ax_client, file_path)
                # Also load for LLM access
                existing_df = pd.read_csv(file_path)
                existing_data_for_llm.append(existing_df)
            else:
                lash_e.logger.info(f"No summary file found in {file_path}, skipping.")
    all_results = []
    raw_measurements = []

    def check_if_measurement_vial_full():
        global state
        current_vial = state["measurement_vial_name"]
        vol = lash_e.nr_robot.get_vial_info(current_vial, "vial_volume")
        if vol > 7.0:
            lash_e.nr_robot.remove_pipet()
            lash_e.nr_robot.return_vial_home(current_vial)
            state["measurement_vial_index"] += 1
            new_vial_name = f"measurement_vial_{state['measurement_vial_index']}"
            state["measurement_vial_name"] = new_vial_name
            lash_e.logger.info(f"[info] Switching to new measurement vial: {new_vial_name}")
            lash_e.nr_robot.move_vial_to_location(new_vial_name, "clamp", 0)


    for i, volume in enumerate(VOLUMES):
        expected_mass = EXPECTED_MASSES[i]
        expected_time = EXPECTED_TIME[i]
        for _ in range(SOBOL_CYCLES_PER_VOLUME):
            params, trial_index = ax_client.get_next_trial()
            check_if_measurement_vial_full()
            result = pipet_and_measure(lash_e, 'liquid_source', state["measurement_vial_name"], volume, params, expected_mass, expected_time, REPLICATES, SIMULATE, autosave_raw_path, raw_measurements, LIQUID, NEW_PIPET_EACH_TIME_SET)
            recommender.add_result(ax_client, trial_index, result)
            result.update(params)
            result.update({"volume": volume, "trial_index": trial_index, "strategy": "SOBOL", "liquid": LIQUID, "time_reported": datetime.now().isoformat()})
            result = strip_tuples(result)
            all_results.append(result)
            if not SIMULATE:
                pd.DataFrame([result]).to_csv(autosave_summary_path, mode='a', index=False, header=not os.path.exists(autosave_summary_path))

    for i, volume in enumerate(VOLUMES):
        expected_mass = EXPECTED_MASSES[i]
        expected_time = EXPECTED_TIME[i]
        for _ in range(BAYES_CYCLES_PER_VOLUME):
            if model_type == 'LLM' and LLM_AVAILABLE:
                # Use LLM to generate parameter suggestions
                config_path = os.path.abspath("recommenders/calibration_unified_config.json")
                print(f"🔍 Looking for config at: {config_path}")
                print(f"🔍 Config file exists: {os.path.exists(config_path)}")
                
                # Prepare LLM input file with existing data + current results
                llm_input_file = os.path.join("output", "temp_llm_input.csv")
                os.makedirs("output", exist_ok=True)  # Ensure output folder exists
                
                # Combine existing data with current results for LLM
                llm_data_frames = []
                if existing_data_for_llm:
                    llm_data_frames.extend(existing_data_for_llm)
                    print(f"🔍 DEBUG: Including {len(existing_data_for_llm)} existing data files for LLM")
                if all_results:
                    llm_data_frames.append(pd.DataFrame(all_results))
                    print(f"🔍 DEBUG: Including {len(all_results)} current results for LLM")
                
                if llm_data_frames:
                    combined_llm_data = pd.concat(llm_data_frames, ignore_index=True)
                    
                    # Limit data size to avoid token limits (keep most recent experiments)
                    MAX_LLM_EXPERIMENTS = 200  # Reduced for gpt-4's 8K token limit
                    if len(combined_llm_data) > MAX_LLM_EXPERIMENTS:
                        combined_llm_data = combined_llm_data.tail(MAX_LLM_EXPERIMENTS)
                        print(f"🔍 DEBUG: Limited to most recent {MAX_LLM_EXPERIMENTS} experiments for LLM")
                    
                    combined_llm_data.to_csv(llm_input_file, index=False)
                    print(f"🔍 DEBUG: Created LLM input file with {len(combined_llm_data)} total experiments")
                else:
                    print("🔍 DEBUG: No data available for LLM - creating empty file")
                    pd.DataFrame().to_csv(llm_input_file, index=False)
                
                # Get LLM recommendations using direct optimizer
                optimizer = llm_opt.LLMOptimizer()
                config = optimizer.load_config(config_path)
                
                # Update config batch_size to match unified BATCH_SIZE
                config["batch_size"] = BATCH_SIZE
                print(f"🔍 DEBUG: Updated LLM config batch_size to {BATCH_SIZE}")
                
                # Specify output file in the output directory to avoid path issues
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                llm_output_file = os.path.join("output", f"llm_recommendations_{timestamp}.csv")
                
                result = optimizer.optimize(llm_input_file, config, llm_output_file)
                
                # Extract parameter suggestions from LLM response
                all_llm_recs = result.get('recommendations', [])
                print(f"🔍 DEBUG: LLM returned {len(all_llm_recs)} total recommendations")
                llm_params_list = all_llm_recs[:BATCH_SIZE]  # Take up to batch size
                print(f"🔍 DEBUG: Taking {len(llm_params_list)} recommendations (batch size: {BATCH_SIZE})")
                llm_summary = result.get('summary', 'No summary provided')
                
                lash_e.logger.info("LLM Optimization Results:")
                lash_e.logger.info(f"Strategy Summary: {llm_summary}")

                # Convert LLM suggestions to ax_client trials
                suggestions = []
                for i, llm_params in enumerate(llm_params_list):
                    print(f"🔍 DEBUG: Processing LLM recommendation {i+1}: {list(llm_params.keys()) if llm_params else 'None'}")
                    if llm_params:  # Make sure we have parameters
                        # Extract metadata fields
                        confidence = llm_params.get('confidence', 'unknown')
                        reasoning = llm_params.get('reasoning', 'No reasoning provided')
                        expected_improvement = llm_params.get('expected_improvement', 'Not specified')
                        
                        # Log LLM rationale and confidence
                        lash_e.logger.info(f"LLM Recommendation {i+1}:")
                        lash_e.logger.info(f"  Confidence Level: {confidence}")
                        lash_e.logger.info(f"  Reasoning: {reasoning}")
                        lash_e.logger.info(f"  Expected Improvement: {expected_improvement}")

                        # Extract only the parameter values that match the search space
                        expected_params = set(ax_client.experiment.search_space.parameters.keys())
                        filtered_params = {k: v for k, v in llm_params.items() if k in expected_params}
                        
                        # Convert parameter types to match ax_client expectations
                        # Float parameters: wait times, retract_speed, air volumes, overaspirate_vol
                        float_params = {'aspirate_wait_time', 'dispense_wait_time', 'retract_speed', 
                                      'pre_asp_air_vol', 'post_asp_air_vol', 'overaspirate_vol'}
                        # Integer parameters: speeds
                        int_params = {'aspirate_speed', 'dispense_speed'}
                        
                        for param_name in filtered_params:
                            if param_name in float_params:
                                filtered_params[param_name] = float(filtered_params[param_name])
                            elif param_name in int_params:
                                filtered_params[param_name] = int(filtered_params[param_name])
                        
                        print(f"🔍 DEBUG: Filtered params for trial {i+1}: {filtered_params}")
                                              
                        params, trial_index = ax_client.attach_trial(filtered_params)
                        suggestions.append((params, trial_index))
                        print(f"🔍 DEBUG: Successfully attached trial {trial_index}")
                
                print(f"🔍 DEBUG: Total suggestions created: {len(suggestions)}")
                
                # Fallback to Bayesian if no LLM suggestions
                if not suggestions:
                    suggestions = recommender.get_suggestions(ax_client, volume, n=BATCH_SIZE)
            else:
                # Use existing Bayesian optimization
                suggestions = recommender.get_suggestions(ax_client, volume, n=BATCH_SIZE)
                
            for params, trial_index in suggestions:
                check_if_measurement_vial_full()
                results = pipet_and_measure(lash_e, 'liquid_source', state["measurement_vial_name"], volume, params, expected_mass, expected_time, REPLICATES, SIMULATE, autosave_raw_path, raw_measurements, LIQUID, NEW_PIPET_EACH_TIME_SET)
                recommender.add_result(ax_client, trial_index, results)
                results.update(params)
                strategy_name = "LLM" if model_type == 'LLM' and LLM_AVAILABLE else "BAYESIAN"
                results.update({"volume": volume, "trial_index": trial_index, "strategy": strategy_name, "liquid": LIQUID, "time_reported": datetime.now().isoformat()})
                results = strip_tuples(results)
                all_results.append(results)
                if not SIMULATE:
                    pd.DataFrame([results]).to_csv(autosave_summary_path, mode='a', index=False, header=not os.path.exists(autosave_summary_path))

    results_df = pd.DataFrame(all_results)
    lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.return_vial_home(state["measurement_vial_name"])
    lash_e.nr_robot.move_home()
    lash_e.logger.info(results_df)

    if not SIMULATE:
        save_analysis(results_df, pd.DataFrame(raw_measurements), autosave_dir)
        slack_agent.send_slack_message(f"Calibration experiment with {LIQUID} and model {model_type} completed. Results saved to {autosave_dir}")
