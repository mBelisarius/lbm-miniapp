import argparse
import itertools
import subprocess
import yaml

from pathlib import Path


def run_simulation():
    """
    Runs LBM simulations for a matrix of pressure ratios and lattice numbers.
    """
    parser = argparse.ArgumentParser(description="Run LBM simulations for different configurations.")
    parser.add_argument("exepath", type=str, help="Path to the LBM simulation executable.")
    parser.add_argument("config", type=str, help="Path to the base config YAML file.")
    parser.add_argument("outpath", type=str, help="Path to the main output directory.")
    parser.add_argument("--plpr", nargs='+', type=float, required=True, help="A list of pressure ratios (pL/pR).")
    parser.add_argument("--nx", nargs='+', type=int, required=True, help="A list of lattice numbers for the x-direction.")

    args = parser.parse_args()

    # Load the base configuration file
    try:
        with open(args.config, 'r') as f:
            base_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Base config file not found at '{args.config}'")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return

    # Extract fluid properties from the base config
    try:
        fluid_props = base_config['Fluid']
        pL = float(fluid_props.get('pressureL', 1.0))
        rhoL = float(fluid_props.get('densityL', 1.0))
        pR_base = float(fluid_props.get('pressureR', 1.0))
        rhoR_base = float(fluid_props.get('densityR', 1.0))
        gamma = float(fluid_props.get('gamma', 1.4))
    except (KeyError, TypeError, ValueError) as e:
        print(f"Error: Invalid or missing fluid properties in the config file. {e}")
        return

    # Ensure the main output directory exists
    Path(args.outpath).mkdir(parents=True, exist_ok=True)

    # Generate all combinations of pressure ratios and nx values
    for plpr_ratio, nx_val in itertools.product(args.plpr, args.nx):
        print(f"--- Processing configuration: pL/pR = {plpr_ratio}, nx = {nx_val} ---")

        # Calculations
        # Calculate right-side pressure
        pR = pL / plpr_ratio

        # Calculate right-side density using the isentropic relation
        # pL / rhoL^gamma = pR / rhoR^gamma  =>  rhoR = rhoL * (pR/pL)^(1/gamma)
        rhoR = rhoR_base * (pR / pR_base) ** (1.0 / gamma)

        print(f"  Calculated pR = {pR:.6f}, rhoR = {rhoR:.6f}")

        # Configuration Setup
        # Create a deep copy to avoid modifying the base config dict
        run_config = base_config.copy()
        run_config['Fluid'] = run_config.get('Fluid', {}).copy()
        run_config['Mesh'] = run_config.get('Mesh', {}).copy()

        # Update the configuration for the current run
        run_config['Fluid']['pressureR'] = pR
        run_config['Fluid']['densityR'] = rhoR
        run_config['Mesh']['nx'] = nx_val

        # File and Directory Setup
        # Create a descriptive name for the run
        run_name = f"plpr_{plpr_ratio}_nx_{nx_val}"
        current_run_dir = Path(args.outpath) / run_name
        current_run_dir.mkdir(exist_ok=True)

        # Path for the new config file
        new_config_path = current_run_dir / "config.yaml"

        # Write the generated configuration to a new YAML file
        with open(new_config_path, 'w') as f:
            yaml.dump(run_config, f, default_flow_style=False, sort_keys=False)

        print(f"  Generated config file at: {new_config_path}")

        # Execute Simulation
        # Build the command to execute the simulation
        command = [
            args.exepath,
            "--config", str(new_config_path),
            "--outpath", str(current_run_dir)
        ]

        print(f"  Executing command: {' '.join(command)}")

        try:
            # Run the simulation
            subprocess.run(command, check=True, capture_output=True, text=True)
            print("  Simulation completed successfully.")
        except FileNotFoundError:
            print(f"  Error: The executable was not found at '{args.exepath}'. Please provide a valid path.")
            # Stop the script if the executable is not found
            break
        except subprocess.CalledProcessError as e:
            print(f"  Error during simulation execution for {run_name}:")
            print(f"  Return code: {e.returncode}")
            print(f"  Stdout: {e.stdout}")
            print(f"  Stderr: {e.stderr}")
        except Exception as e:
            print(f"  An unexpected error occurred: {e}")


if __name__ == "__main__":
    run_simulation()
