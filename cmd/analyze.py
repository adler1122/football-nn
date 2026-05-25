from config import Config
import utils
from tracking import Tracker
from assigner import Assigner

def parse_args(argv, config: Config):
    """
    Docstring for parse_args
    
    :param argv: Description
    :param config: Description
    :type config: Config
    """

    return

def run_analyzer(args, config: Config):
    """
    Docstring for analyze
    
    :param args: Description
    :param config: Description
    :type config: Config
    """

    video_frames = utils.read_video(config.input_video_path)

    tracker = Tracker(config.Analyzer.player_model_path, config.device)
    tracks = tracker.track_detections(video_frames)

    assigner = Assigner()
    assigner.assign_team(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        # assigner.assign_team(video_frames[frame_num], tracks['players'][frame_num])
        for player_id, track in player_track.items():
            team = assigner.get_player_team(video_frames[frame_num],
                                                 track['bounding_box'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            if team == 1:
                tracks['players'][frame_num][player_id]['team_color'] = config.colors[1]
            else:
                tracks['players'][frame_num][player_id]['team_color'] = config.colors[2]

    output_video_frames = tracker.draw_annotations(video_frames,tracks)
    utils.save_video(output_video_frames,config.output_video_path)
