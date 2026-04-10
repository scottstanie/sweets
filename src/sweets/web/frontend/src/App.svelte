<script>
  import Map from './lib/Map.svelte'
  import JobList from './lib/JobList.svelte'
  import ConfigForm from './lib/ConfigForm.svelte'
  import LogViewer from './lib/LogViewer.svelte'

  let selectedBbox = $state(null)
  let showConfigForm = $state(false)
  let viewingLogsJobId = $state(null)

  function handleBboxSelected(event) {
    selectedBbox = event.detail
    showConfigForm = true
  }

  function handleJobCreated() {
    showConfigForm = false
    selectedBbox = null
  }

  function handleViewLogs(jobId) {
    viewingLogsJobId = jobId
  }

  function closeLogViewer() {
    viewingLogsJobId = null
  }
</script>

<main>
  <header>
    <h1>Sweets</h1>
    <p>InSAR Workflow Manager</p>
  </header>

  <div class="container">
    <div class="map-panel">
      <Map onBboxSelected={handleBboxSelected} />
      {#if showConfigForm && selectedBbox}
        <div class="config-overlay">
          <ConfigForm
            bbox={selectedBbox}
            onClose={() => showConfigForm = false}
            onJobCreated={handleJobCreated}
          />
        </div>
      {/if}
    </div>

    <aside class="sidebar">
      {#if viewingLogsJobId}
        <LogViewer jobId={viewingLogsJobId} onClose={closeLogViewer} />
      {:else}
        <JobList onViewLogs={handleViewLogs} />
      {/if}
    </aside>
  </div>
</main>

<style>
  :global(body) {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #1a1a2e;
    color: #eee;
  }

  main {
    display: flex;
    flex-direction: column;
    height: 100vh;
  }

  header {
    padding: 0.5rem 1rem;
    background: #16213e;
    border-bottom: 1px solid #0f3460;
  }

  header h1 {
    margin: 0;
    font-size: 1.25rem;
    color: #e94560;
  }

  header p {
    margin: 0;
    font-size: 0.75rem;
    color: #888;
  }

  .container {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  .map-panel {
    flex: 1;
    position: relative;
  }

  .config-overlay {
    position: absolute;
    top: 1rem;
    left: 1rem;
    z-index: 10;
    background: #16213e;
    border-radius: 8px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    max-width: 400px;
    max-height: calc(100% - 2rem);
    overflow-y: auto;
  }

  .sidebar {
    width: 320px;
    background: #16213e;
    border-left: 1px solid #0f3460;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
  }
</style>
