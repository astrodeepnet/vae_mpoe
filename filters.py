def fetch_filter_data(filters):
    filter_dict = {}
    for name, filepath in filters:
        try:
            with open(filepath, 'r') as file:
                wavelength, transmission = [], []
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        try:
                            wavelength.append(float(parts[0]))
                            transmission.append(float(parts[1]))
                        except ValueError:
                            continue  # Skip malformed lines
                filter_dict[name] = {"wl": wavelength, "tr": transmission}
        except FileNotFoundError:
            print(f"File not found for {name}: {filepath}")
        except Exception as e:
            print(f"Error reading file for {name}: {e}")
    return filter_dict

# Define filters and load them
filters = [
    ['HSCg', 'HSC_bands/HSC.g_filter.dat'],
    ['HSCr', 'HSC_bands/HSC.r_filter.dat'],
    ['HSCi', 'HSC_bands/HSC.i_filter.dat'],
    ['HSCz', 'HSC_bands/HSC.z_filter.dat'],
    ['HSCY', 'HSC_bands/HSC.Y_filter.dat']
]

filter_data = fetch_filter_data(filters)